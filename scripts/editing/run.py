"""Core editing logic for API-based response editing with quality tracking."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path

from datasets import Dataset

from scripts.config import PipelineConfig
from scripts.editing import anthropic_client, openai_client
from scripts.editing.prompts import get_prompt
from scripts.editing.quality import (
    EditQualityMetric,
    QualityReporter,
    aggregate_metrics,
    get_metric,
    get_reporters,
)
from scripts.utils import read_jsonl, write_jsonl, setup_logging


TokenUsage = dict[str, int]


def validate_api_key(provider: str) -> None:
    """Validate that the required API key is set for the provider."""
    provider = provider.lower()
    if provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is required for Anthropic provider"
            )
    elif provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY environment variable is required for OpenAI provider"
            )
    else:
        raise ValueError(f"Unsupported editing provider: {provider}")


def ensure_run_id(config: PipelineConfig) -> str:
    """Generate a run ID if not set."""
    if config.run_id:
        return config.run_id
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = f"{timestamp}-editing"
    config.run_id = run_id
    return run_id


def select_client(provider: str):
    """Select the appropriate API client based on provider."""
    provider = provider.lower()
    if provider == "anthropic":
        return anthropic_client.edit_response
    if provider == "openai":
        return openai_client.edit_response
    raise ValueError(f"Unsupported editing provider: {provider}")


def accumulate_usage(total: TokenUsage, usage: TokenUsage | None) -> None:
    """Accumulate token usage from a single request."""
    if not usage:
        return
    total["input_tokens"] += usage.get("input_tokens", 0)
    total["output_tokens"] += usage.get("output_tokens", 0)
    total["total_tokens"] += usage.get("total_tokens", 0)


def init_quality_metrics(config: PipelineConfig) -> list[EditQualityMetric]:
    """Initialize quality metrics from config."""
    if not config.editing.quality.enabled:
        return []
    return [get_metric(name) for name in config.editing.quality.metrics]


def init_quality_reporters(config: PipelineConfig) -> list[QualityReporter]:
    """Initialize quality reporters from config."""
    if not config.editing.quality.enabled:
        return []
    return get_reporters(config.editing.quality.reporters)


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


async def edit_dataset(records: list[dict[str, str]], config: PipelineConfig) -> list[dict]:
    """Edit all records asynchronously with rate limiting."""
    logger = setup_logging()
    max_concurrent = max(1, config.editing.max_concurrent)
    semaphore = asyncio.Semaphore(max_concurrent)
    edit_func = select_client(config.editing.provider)

    # Initialize quality metrics and reporters
    quality_metrics = init_quality_metrics(config)
    quality_reporters = init_quality_reporters(config)
    metrics_key = config.editing.quality.metrics_key
    all_record_metrics: list[dict[str, float | int]] = []

    async def edit_one(index: int, record: dict[str, str]):
        async with semaphore:
            prompt = get_prompt(
                config.editing.prompt_template,
                question=record["question"],
                response=record["response"],
            )
            edited_text, usage = await edit_func(prompt, config.editing)
            return index, edited_text, usage

    tasks = [asyncio.create_task(edit_one(i, record)) for i, record in enumerate(records)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    edited_records: list[dict] = []
    total_usage: TokenUsage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    failed_count = 0

    for result in results:
        if isinstance(result, Exception):
            failed_count += 1
            logger.warning("Editing failed: %s", result)
            continue
        index, edited_text, usage = result
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

    # Aggregate and report summary metrics
    if quality_metrics and all_record_metrics:
        aggregates = aggregate_metrics(all_record_metrics)
        for reporter in quality_reporters:
            reporter.report_summary(aggregates)

    logger.info(
        "Editing complete: %d succeeded, %d failed. Tokens: %d input, %d output",
        len(edited_records),
        failed_count,
        total_usage["input_tokens"],
        total_usage["output_tokens"],
    )

    return edited_records


def run_editing(config: PipelineConfig, dataset: Dataset | None = None) -> Dataset:
    """Edit model responses using an LLM API.

    Args:
        config: Pipeline configuration.
        dataset: Optional pre-loaded inference output. If None, loads from config paths.

    Returns:
        Dataset with added 'edited_response' column.

    Raises:
        ValueError: If the required API key is not set or all editing requests fail.
    """
    logger = setup_logging()
    run_id = ensure_run_id(config)

    validate_api_key(config.editing.provider)

    if dataset is None:
        input_path = Path(config.inference.output.save_path.format(run_id=run_id))
        if not input_path.exists():
            raise FileNotFoundError(f"Inference output not found: {input_path}")
        records = read_jsonl(input_path)
        dataset = Dataset.from_list(records)

    required = {"question", "response"}
    missing = required.difference(dataset.column_names)
    if missing:
        raise ValueError(f"Editing dataset missing columns: {sorted(missing)}")

    records = dataset.to_list()
    edited_records = asyncio.run(edit_dataset(records, config))

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

    result = Dataset.from_list(edited_records)
    save_path = Path(config.editing.output.save_path.format(run_id=run_id))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(result.to_list(), save_path)
    logger.info("Saved edited dataset to %s", save_path)
    return result
