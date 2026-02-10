"""Core editing logic for API-based response editing with quality tracking."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from datasets import Dataset

from scripts.common.config import GenerationConfig
from scripts.editing.config import EditingConfig, EditingResult
from scripts.editing.prompts import get_prompt
from scripts.editing.quality import (
    EditQualityMetric,
    QualityReporter,
    aggregate_metrics,
    get_metric,
    get_reporters,
)
from scripts.inference.config import (
    InferenceConfig,
    RetryConfig as InferenceRetryConfig,
    AnthropicProviderConfig as InferenceAnthropicProviderConfig,
)
from scripts.inference.providers.base import InferenceProvider, accumulate_usage
from scripts.inference.providers import get_provider
from scripts.utils import read_jsonl, write_jsonl, setup_logging


TokenUsage = dict[str, int]


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
    else:
        max_tokens = config.anthropic.max_tokens
        anthropic_cfg = InferenceAnthropicProviderConfig(
            max_tokens=config.anthropic.max_tokens
        )

    # Match provider defaults since EditingConfig doesn't expose sampling params.
    generation = GenerationConfig(
        max_new_tokens=max_tokens,
        temperature=1.0,
        top_p=1.0,
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
        anthropic=anthropic_cfg,
    )


def init_quality_metrics(config: EditingConfig) -> list[EditQualityMetric]:
    """Initialize quality metrics from config."""
    if not config.quality.enabled:
        return []
    return [get_metric(name) for name in config.quality.metrics]


def init_quality_reporters(config: EditingConfig) -> list[QualityReporter]:
    """Initialize quality reporters from config."""
    if not config.quality.enabled:
        return []
    return get_reporters(config.quality.reporters)


def compute_record_metrics(
    metrics: list[EditQualityMetric], original: str, edited: str
) -> dict[str, float | int]:
    """Compute all metrics for a single record."""
    result: dict[str, float | int] = {}
    for metric in metrics:
        result.update(metric.compute(original, edited))
    return result


def apply_reporters(
    reporters: list[QualityReporter],
    record: dict,
    metrics_values: dict[str, float | int],
    metrics_key: str,
) -> dict:
    """Apply all reporters to a record."""
    for reporter in reporters:
        record = reporter.report_record(record, metrics_values, metrics_key)
    return record


async def edit_dataset(
    records: list[dict[str, str]],
    config: EditingConfig,
    provider: InferenceProvider,
    save_path: Path | None = None,
) -> tuple[list[dict], TokenUsage, int]:
    """Edit all records asynchronously with rate limiting.

    Results are written incrementally to save_path (one JSONL row per completion)
    so that partial progress survives crashes.

    Returns:
        Tuple of (edited_records, total_usage, failed_count).
    """
    logger = setup_logging()
    total = len(records)
    max_concurrent = max(1, config.max_concurrent)
    model_name = getattr(provider, "model", config.model)

    # Initialize quality metrics and reporters
    quality_metrics = init_quality_metrics(config)
    quality_reporters = init_quality_reporters(config)
    metrics_key = config.quality.metrics_key
    all_record_metrics: list[dict[str, float | int]] = []

    edited_records: list[dict] = []
    total_usage: TokenUsage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    succeeded_count = 0
    failed_count = 0

    # Prepare incremental output file
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        out_file = open(save_path, "w")
    else:
        out_file = None

    semaphore = asyncio.Semaphore(max_concurrent)

    async def edit_one(index: int, record: dict[str, str]):
        async with semaphore:
            prompt = get_prompt(
                config.prompt_template,
                question=record["question"],
                response=record["response"],
            )
            # Print what is sent to the teacher
            print(f"\n{'='*60}")
            print(f"PROMPT SENT TO TEACHER (sample {index+1}/{total})")
            print(f"Model: {model_name}")
            print(f"{'='*60}")
            print(prompt)
            print(f"{'='*60}\n")

            responses, usage, _ = await provider.generate_batch_with_metadata_async(
                [prompt], num_responses=1
            )
            if len(responses) != 1:
                raise ValueError(
                    "Provider returned unexpected number of responses. "
                    f"Expected 1, got {len(responses)}."
                )
            return index, responses[0], usage

    tasks = [asyncio.create_task(edit_one(i, record)) for i, record in enumerate(records)]

    try:
        for coro in asyncio.as_completed(tasks):
            try:
                index, edited_text, usage = await coro
            except Exception as exc:
                failed_count += 1
                logger.warning("Editing failed for a sample: %s", exc)
                logger.info(
                    "Progress: %d/%d done (%d succeeded, %d failed)",
                    succeeded_count + failed_count,
                    total,
                    succeeded_count,
                    failed_count,
                )
                continue

            record = dict(records[index])
            record["edited_response"] = edited_text

            # Compute quality metrics for this record
            if quality_metrics:
                metrics_values = compute_record_metrics(
                    quality_metrics, record["response"], edited_text
                )
                all_record_metrics.append(metrics_values)
                record = apply_reporters(
                    quality_reporters, record, metrics_values, metrics_key
                )

            edited_records.append(record)
            accumulate_usage(total_usage, usage)
            succeeded_count += 1

            # Write row immediately so progress is saved to disk
            if out_file is not None:
                out_file.write(json.dumps(record) + "\n")
                out_file.flush()

            # Log progress every row
            if succeeded_count % 10 == 0 or succeeded_count == total - failed_count:
                logger.info(
                    "Progress: %d/%d done (%d succeeded, %d failed, %d in-flight)",
                    succeeded_count + failed_count,
                    total,
                    succeeded_count,
                    failed_count,
                    total - succeeded_count - failed_count,
                )
    finally:
        if out_file is not None:
            out_file.close()

    # Aggregate and report summary metrics
    if quality_metrics and all_record_metrics:
        aggregates = aggregate_metrics(all_record_metrics)
        for reporter in quality_reporters:
            reporter.report_summary(aggregates)

    logger.info(
        "Editing complete: %d/%d succeeded, %d failed. Tokens: %d input, %d output",
        succeeded_count,
        total,
        failed_count,
        total_usage["input_tokens"],
        total_usage["output_tokens"],
    )

    return edited_records, total_usage, failed_count


def run_editing(
    config: EditingConfig,
    dataset: Dataset | None = None,
    input_path: Path | None = None,
) -> tuple[Dataset, EditingResult]:
    """Edit model responses using an LLM API.

    Args:
        config: Editing configuration.
        dataset: Optional pre-loaded inference output dataset.
        input_path: Optional path to load input dataset from (if dataset is None).

    Returns:
        Tuple of (dataset with 'edited_response' column, EditingResult metadata).

    Raises:
        ValueError: If the required API key is not set or all editing requests fail.

    Example:
        config = EditingConfig(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            output_path=Path("scratch/edited.jsonl"),
        )
        dataset, result = run_editing(config, input_dataset)
    """
    logger = setup_logging()

    if dataset is None:
        if input_path is None:
            raise ValueError("Either dataset or input_path must be provided")
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        records = read_jsonl(input_path)
        dataset = Dataset.from_list(records)

    required = {"question", "response"}
    missing = required.difference(dataset.column_names)
    if missing:
        raise ValueError(f"Editing dataset missing columns: {sorted(missing)}")

    records = dataset.to_list()
    inference_config = build_inference_config(config)
    provider = get_provider(inference_config.provider, inference_config)
    edited_records, total_usage, failed_count = asyncio.run(
        edit_dataset(records, config, provider)
    )

    if not edited_records:
        raise ValueError(
            f"All {len(records)} editing requests failed. Check API key and provider configuration."
        )

    if len(edited_records) < len(records):
        logger.warning(
            "Some editing requests failed: %d/%d succeeded",
            len(edited_records),
            len(records),
        )

    result_dataset = Dataset.from_list(edited_records)

    # Create result metadata
    result = EditingResult(
        num_samples=len(edited_records),
        num_failed=failed_count,
        total_input_tokens=total_usage["input_tokens"],
        total_output_tokens=total_usage["output_tokens"],
    )

    # Save if output path specified
    if config.output_path:
        save_path = Path(config.output_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(result_dataset.to_list(), save_path)
        logger.info("Saved edited dataset to %s", save_path)
        result.output_path = save_path

    return result_dataset, result
