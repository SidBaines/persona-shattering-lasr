"""Core editing logic for LLM- or code-based response editing with quality tracking."""

from __future__ import annotations

import asyncio
import importlib
import json
from numbers import Real
from pathlib import Path
from typing import Any, Callable, TextIO

from datasets import Dataset

from scripts.common.config import GenerationConfig
from scripts.common.persona_registry import get_persona_default_evaluations
from scripts.editing.config import EditingConfig, EditingResult
from scripts.editing.prompts import get_prompt
from scripts.persona_metrics import (
    PersonaMetricsConfig,
    PersonaMetricSpec,
    aggregate_persona_metric_results,
    run_persona_metrics,
)
from scripts.inference.config import (
    AnthropicProviderConfig as InferenceAnthropicProviderConfig,
)
from scripts.inference.config import InferenceConfig
from scripts.inference.config import OpenAIProviderConfig as InferenceOpenAIProviderConfig
from scripts.inference.config import RetryConfig as InferenceRetryConfig
from scripts.inference.providers import get_provider
from scripts.inference.providers.base import InferenceProvider, accumulate_usage
from scripts.utils import (
    count_jsonl_rows,
    iter_jsonl_batches,
    read_jsonl,
    setup_logging,
    write_jsonl,
)

TokenUsage = dict[str, int]
MetricValue = float | int | str

# Progress logging frequency: log every N successful edits
PROGRESS_LOG_INTERVAL = 10


def load_code_editor(editor_path: str) -> Callable[[str, dict], str]:
    """Load a code-based editor from a 'module.submodule:func' path."""
    if ":" not in editor_path:
        raise ValueError(
            "Code editor path must be in the form 'module.submodule:func'. "
            f"Got: {editor_path}"
        )
    module_path, attr = editor_path.split(":", 1)
    module = importlib.import_module(module_path)
    editor = getattr(module, attr, None)
    if editor is None:
        raise ValueError(f"Editor '{attr}' not found in module '{module_path}'.")
    if not callable(editor):
        raise TypeError(f"Editor '{editor_path}' is not callable.")
    return editor


def build_inference_config(config: EditingConfig) -> InferenceConfig:
    """Create an InferenceConfig from an EditingConfig."""
    provider = config.provider.lower()
    if provider not in {"openai", "anthropic"}:
        raise ValueError(f"Unsupported editing provider: {provider}")

    model = config.model
    if provider == "openai" and config.openai.model:
        model = config.openai.model

    if provider == "openai":
        max_tokens = config.openai.max_tokens
        anthropic_cfg = InferenceAnthropicProviderConfig()
        openai_cfg = InferenceOpenAIProviderConfig(
            reasoning_effort=config.openai.reasoning_effort
        )
    else:
        max_tokens = config.anthropic.max_tokens
        anthropic_cfg = InferenceAnthropicProviderConfig(
            max_tokens=config.anthropic.max_tokens
        )
        openai_cfg = InferenceOpenAIProviderConfig()

    generation = GenerationConfig(
        max_new_tokens=max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        do_sample=True,
        batch_size=max(1, config.max_concurrent),
        num_responses_per_prompt=1,
    )

    return InferenceConfig(
        model=model,
        provider=provider,
        generation=generation,
        max_concurrent=config.max_concurrent,
        timeout=config.timeout,
        retry=InferenceRetryConfig(
            max_retries=config.retry.max_retries,
            backoff_factor=config.retry.backoff_factor,
        ),
        # Let per-record tasks surface errors so we can count failures.
        continue_on_error=False,
        log_failures=True,
        openai=openai_cfg,
        anthropic=anthropic_cfg,
    )


def _resolve_quality_evaluations(config: EditingConfig) -> list[str | PersonaMetricSpec]:
    """Resolve quality evaluations from config.

    When ``config.quality.evaluations`` is ``None`` (the default), the
    evaluation is auto-resolved from the active persona.
    """
    if config.quality.evaluations is not None:
        return list(config.quality.evaluations)
    return get_persona_default_evaluations(config.quality.persona)


def _build_quality_comparison_metrics(
    original_metrics: dict[str, MetricValue],
    edited_metrics: dict[str, MetricValue],
) -> dict[str, MetricValue]:
    """Build quality metrics by comparing original vs edited evaluation outputs."""
    metrics: dict[str, MetricValue] = {}
    all_metric_keys = sorted(set(original_metrics.keys()) | set(edited_metrics.keys()))

    for metric_key in all_metric_keys:
        if metric_key in original_metrics:
            metrics[f"{metric_key}.original"] = original_metrics[metric_key]
        if metric_key in edited_metrics:
            metrics[f"{metric_key}.edited"] = edited_metrics[metric_key]

        original_value = original_metrics.get(metric_key)
        edited_value = edited_metrics.get(metric_key)
        if isinstance(original_value, Real) and isinstance(edited_value, Real):
            metrics[f"{metric_key}.delta"] = float(edited_value) - float(original_value)

    return metrics


def _run_quality_evaluation_pass(
    dataset: Dataset,
    config: EditingConfig,
) -> tuple[Dataset, dict[str, object]]:
    """Run configured evaluations on original and edited responses and compare them."""
    if not config.quality.enabled:
        return dataset, {}

    evaluations = _resolve_quality_evaluations(config)
    if not evaluations:
        return dataset, {}

    question_column = "question" if "question" in dataset.column_names else None
    original_key = "_quality_original_metrics"
    edited_key = "_quality_edited_metrics"

    original_eval_config = PersonaMetricsConfig(
        evaluations=evaluations,
        response_column="response",
        question_column=question_column,
        judge=config.quality.judge,
        metrics_key=original_key,
    )
    with_original_metrics, _ = run_persona_metrics(original_eval_config, dataset=dataset)

    edited_eval_config = PersonaMetricsConfig(
        evaluations=evaluations,
        response_column="edited_response",
        question_column=question_column,
        judge=config.quality.judge,
        metrics_key=edited_key,
    )
    with_all_metrics, _ = run_persona_metrics(edited_eval_config, dataset=with_original_metrics)

    records = with_all_metrics.to_list()
    all_quality_metrics: list[dict[str, MetricValue]] = []
    for record in records:
        original_metrics = record.pop(original_key, {})
        edited_metrics = record.pop(edited_key, {})
        if not isinstance(original_metrics, dict):
            original_metrics = {}
        if not isinstance(edited_metrics, dict):
            edited_metrics = {}

        quality_metrics = _build_quality_comparison_metrics(
            original_metrics,
            edited_metrics,
        )
        existing = record.get(config.quality.metrics_key)
        if isinstance(existing, dict):
            record[config.quality.metrics_key] = {**existing, **quality_metrics}
        else:
            record[config.quality.metrics_key] = quality_metrics
        all_quality_metrics.append(quality_metrics)

    return Dataset.from_list(records), aggregate_persona_metric_results(all_quality_metrics)


async def edit_dataset(
    records: list[dict[str, str]],
    config: EditingConfig,
    provider: InferenceProvider,
    out_file: TextIO | None = None,
    index_offset: int = 0,
) -> tuple[list[dict], TokenUsage, int]:
    """Edit a batch of records asynchronously with ordered writes."""
    logger = setup_logging()
    total = len(records)
    max_concurrent = max(1, config.max_concurrent)

    edited_records: list[dict] = []
    total_usage: TokenUsage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    succeeded_count = 0
    failed_count = 0

    semaphore = asyncio.Semaphore(max_concurrent)

    async def edit_one(
        local_index: int, record: dict[str, str]
    ) -> tuple[int, str, TokenUsage, Exception | None]:
        try:
            async with semaphore:
                prompt = get_prompt(
                    config.prompt_template,
                    question=record["question"],
                    response=record["response"],
                )

                responses, usage, _ = await provider.generate_batch_with_metadata_async(
                    [prompt], num_responses=1
                )
                if len(responses) != 1:
                    raise ValueError(
                        "Provider returned unexpected number of responses. "
                        f"Expected 1, got {len(responses)}."
                    )
                return local_index, responses[0], usage, None
        except Exception as exc:  # noqa: BLE001
            return local_index, "", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}, exc

    tasks = [asyncio.create_task(edit_one(i, record)) for i, record in enumerate(records)]
    ready: dict[int, tuple[str, TokenUsage]] = {}
    failed_indexes: set[int] = set()
    next_to_emit = 0

    for task in asyncio.as_completed(tasks):
        done_before = succeeded_count + failed_count
        local_index, edited_text, usage, error = await task
        if error is None:
            ready[local_index] = (edited_text, usage)
        else:
            failed_count += 1
            failed_indexes.add(local_index)
            logger.warning("Editing failed for a sample: %s", error)

        while next_to_emit < total:
            if next_to_emit in failed_indexes:
                failed_indexes.remove(next_to_emit)
                next_to_emit += 1
                continue
            if next_to_emit not in ready:
                break

            edited_text, usage = ready.pop(next_to_emit)
            result_record = dict(records[next_to_emit])
            result_record["input_index"] = index_offset + next_to_emit
            result_record["edited_response"] = edited_text
            edited_records.append(result_record)
            accumulate_usage(total_usage, usage)
            succeeded_count += 1
            if out_file is not None:
                out_file.write(json.dumps(result_record) + "\n")
                out_file.flush()
            next_to_emit += 1

        done_count = succeeded_count + failed_count
        if done_count != done_before and (
            succeeded_count % PROGRESS_LOG_INTERVAL == 0 or done_count == total
        ):
            logger.info(
                "Progress: %d/%d done (%d succeeded, %d failed, %d in-flight)",
                done_count,
                total,
                succeeded_count,
                failed_count,
                total - done_count,
            )

    logger.info(
        "Editing complete: %d/%d succeeded, %d failed. Tokens: %d input, %d output",
        succeeded_count,
        total,
        failed_count,
        total_usage["input_tokens"],
        total_usage["output_tokens"],
    )

    return edited_records, total_usage, failed_count


def edit_dataset_with_code(
    records: list[dict[str, str]],
    editor: Callable[[str, dict], str],
    out_file: TextIO | None = None,
    index_offset: int = 0,
) -> tuple[list[dict], TokenUsage, int]:
    """Edit a batch of records with a local code-based editor."""
    logger = setup_logging()
    total = len(records)

    edited_records: list[dict] = []
    total_usage: TokenUsage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    succeeded_count = 0
    failed_count = 0

    for row_offset, record in enumerate(records):
        try:
            edited_text = editor(record["response"], record)
            if not isinstance(edited_text, str):
                raise TypeError(
                    "Code editor must return a string. "
                    f"Got {type(edited_text).__name__}."
                )
        except Exception as exc:
            failed_count += 1
            logger.warning("Code editing failed for a sample: %s", exc)
            continue

        result_record = dict(record)
        result_record["input_index"] = index_offset + row_offset
        result_record["edited_response"] = edited_text
        edited_records.append(result_record)
        succeeded_count += 1

        if out_file is not None:
            out_file.write(json.dumps(result_record) + "\n")
            out_file.flush()

        if succeeded_count % PROGRESS_LOG_INTERVAL == 0 or succeeded_count + failed_count == total:
            logger.info(
                "Progress: %d/%d done (%d succeeded, %d failed)",
                succeeded_count + failed_count,
                total,
                succeeded_count,
                failed_count,
            )

    logger.info(
        "Editing complete: %d/%d succeeded, %d failed.",
        succeeded_count,
        total,
        failed_count,
    )

    return edited_records, total_usage, failed_count


def _validate_required_columns(records: list[dict[str, Any]], *, context: str) -> None:
    required = {"question", "response"}
    for i, record in enumerate(records):
        missing = [key for key in required if key not in record]
        if missing:
            raise ValueError(
                f"Editing dataset missing columns {missing} at {context} record index {i}."
            )


def run_editing(
    config: EditingConfig,
    dataset: Dataset | None = None,
    input_path: Path | None = None,
) -> tuple[Dataset, EditingResult]:
    """Edit model responses using an LLM API or a code-based editor."""
    logger = setup_logging()

    io_batch_size = max(1, config.io_batch_size)
    provider_name = config.provider.lower()
    if provider_name not in {"anthropic", "openai", "code"}:
        raise ValueError(f"Unsupported editing provider: {config.provider}")

    if dataset is None:
        if input_path is None:
            raise ValueError("Either dataset or input_path must be provided")
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        total_input_rows = count_jsonl_rows(input_path)
    else:
        required = {"question", "response"}
        missing = required.difference(dataset.column_names)
        if missing:
            raise ValueError(f"Editing dataset missing columns: {sorted(missing)}")
        total_input_rows = len(dataset)

    save_path = Path(config.output_path) if config.output_path else None
    resumed_records: list[dict[str, Any]] = []
    resume_rows = 0
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if config.overwrite_output:
            save_path.write_text("")
        elif config.resume and save_path.exists():
            resumed_records = read_jsonl(save_path)
            if resumed_records:
                last_record = resumed_records[-1]
                input_index = last_record.get("input_index")
                if isinstance(input_index, int) and input_index >= 0:
                    resume_rows = input_index + 1
                else:
                    resume_rows = len(resumed_records)
            if resume_rows > total_input_rows:
                raise ValueError(
                    f"Output already has {resume_rows} rows, but input has only "
                    f"{total_input_rows} rows."
                )
            if resume_rows:
                logger.info(
                    "Resuming editing from row %d/%d using existing output at %s",
                    resume_rows,
                    total_input_rows,
                    save_path,
                )
        else:
            save_path.write_text("")

    editor: Callable[[str, dict], str] | None = None
    provider: InferenceProvider | None = None
    if provider_name == "code":
        editor = load_code_editor(config.code.editor)
    else:
        inference_config = build_inference_config(config)
        provider = get_provider(inference_config.provider, inference_config)

    edited_records: list[dict[str, Any]] = list(resumed_records)
    total_usage: TokenUsage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    failed_count = 0

    out_file = (
        save_path.open("a", encoding="utf-8")
        if save_path is not None and resume_rows < total_input_rows
        else None
    )
    try:
        if resume_rows < total_input_rows:
            if dataset is not None:
                records = dataset.to_list()
                for start in range(resume_rows, len(records), io_batch_size):
                    batch = records[start : start + io_batch_size]
                    _validate_required_columns(batch, context=f"dataset[{start}]")
                    if provider_name == "code":
                        assert editor is not None
                        batch_edited, batch_usage, batch_failed = edit_dataset_with_code(
                            batch, editor, out_file=out_file, index_offset=start
                        )
                    else:
                        assert provider is not None
                        batch_edited, batch_usage, batch_failed = asyncio.run(
                            edit_dataset(
                                batch,
                                config,
                                provider,
                                out_file=out_file,
                                index_offset=start,
                            )
                        )
                    edited_records.extend(batch_edited)
                    accumulate_usage(total_usage, batch_usage)
                    failed_count += batch_failed
            else:
                assert input_path is not None
                current_start = resume_rows
                for batch_num, batch in enumerate(
                    iter_jsonl_batches(input_path, io_batch_size, skip_rows=resume_rows),
                    start=1,
                ):
                    _validate_required_columns(batch, context=f"input batch {batch_num}")
                    if provider_name == "code":
                        assert editor is not None
                        batch_edited, batch_usage, batch_failed = edit_dataset_with_code(
                            batch,
                            editor,
                            out_file=out_file,
                            index_offset=current_start,
                        )
                    else:
                        assert provider is not None
                        batch_edited, batch_usage, batch_failed = asyncio.run(
                            edit_dataset(
                                batch,
                                config,
                                provider,
                                out_file=out_file,
                                index_offset=current_start,
                            )
                        )
                    edited_records.extend(batch_edited)
                    accumulate_usage(total_usage, batch_usage)
                    failed_count += batch_failed
                    current_start += len(batch)
        else:
            logger.info("Output already complete. Skipping editing generation.")
    finally:
        if out_file is not None:
            out_file.close()

    if not edited_records:
        raise ValueError(
            f"All {total_input_rows} editing requests failed. "
            "Check provider configuration and credentials."
        )

    if len(edited_records) < total_input_rows:
        logger.warning(
            "Some editing requests failed: %d/%d succeeded",
            len(edited_records),
            total_input_rows,
        )

    result_dataset = Dataset.from_list(edited_records)

    # Run post-edit quality evaluations through the shared evaluation module.
    quality_aggregates: dict[str, object] = {}
    quality_error: str | None = None
    if config.quality.enabled:
        try:
            result_dataset, quality_aggregates = _run_quality_evaluation_pass(
                result_dataset, config
            )
            if quality_aggregates:
                logger.info("Quality evaluation summary:")
                for key, value in sorted(quality_aggregates.items()):
                    if isinstance(value, float):
                        logger.info("  %s: %.4f", key, value)
                    else:
                        logger.info("  %s: %s", key, value)
        except Exception as exc:
            quality_error = (
                "Post-edit quality evaluation failed after edits were generated. "
                f"{type(exc).__name__}: {exc}"
            )
            if config.quality.on_error == "raise":
                raise RuntimeError(
                    f"{quality_error} "
                    "Set `quality.on_error='warn'` or use `--quality-on-error warn` "
                    "to keep edited outputs without quality metrics."
                ) from exc
            logger.warning("%s", quality_error)
            logger.warning(
                "Continuing without quality metrics. "
                "Use `--quality-on-error raise` to fail hard instead."
            )

    # Create result metadata
    result = EditingResult(
        num_samples=len(result_dataset),
        num_failed=failed_count,
        total_input_tokens=total_usage["input_tokens"],
        total_output_tokens=total_usage["output_tokens"],
        quality_error=quality_error,
    )

    # Save final dataset (ensures quality fields are persisted if enabled)
    if save_path is not None:
        write_jsonl(result_dataset.to_list(), save_path)
        logger.info("Saved edited dataset to %s", save_path)
        result.output_path = save_path

    return result_dataset, result
