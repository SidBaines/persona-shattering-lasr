"""Core persona metric logic for running metrics on datasets."""

from __future__ import annotations

import asyncio
from pathlib import Path

from datasets import Dataset

from scripts.data_loading import load_dataset_from_config
from scripts.persona_metrics.aggregation import aggregate_persona_metric_results
from scripts.persona_metrics.base import PersonaMetric, PersonaMetricContext
from scripts.persona_metrics.config import (
    PersonaMetricsConfig,
    PersonaMetricsResult,
    PersonaMetricSpec,
)
from scripts.persona_metrics.registry import get_persona_metric
from scripts.utils import setup_logging, write_jsonl


def create_persona_metrics(config: PersonaMetricsConfig) -> list[PersonaMetric]:
    """Create persona metric instances from config.

    Supports both plain string names (use global judge config) and
    PersonaMetricSpec objects (merge global judge config with per-eval params).
    """
    metrics: list[PersonaMetric] = []
    for spec in config.evaluations:
        if isinstance(spec, str):
            metrics.append(get_persona_metric(spec, judge_config=config.judge))
        else:
            kwargs: dict = {"judge_config": config.judge}
            kwargs.update(spec.params)
            metrics.append(get_persona_metric(spec.name, **kwargs))
    return metrics


async def run_persona_metrics_async(
    config: PersonaMetricsConfig, dataset: Dataset | None = None
) -> tuple[Dataset, PersonaMetricsResult]:
    """Run persona metrics on a dataset asynchronously.

    Args:
        config: Persona metrics configuration.
        dataset: Optional pre-loaded dataset. If None, loads from config.dataset.

    Returns:
        Tuple of (dataset with persona metrics added, PersonaMetricsResult metadata).
    """
    logger = setup_logging()

    # Load dataset if not provided
    if dataset is None:
        dataset = load_dataset_from_config(config.dataset)

    # Validate required columns
    if config.response_column not in dataset.column_names:
        raise ValueError(
            f"Dataset missing response column '{config.response_column}'. "
            f"Available columns: {dataset.column_names}"
        )

    # Extract responses and questions
    responses = dataset[config.response_column]
    questions = None
    if config.question_column and config.question_column in dataset.column_names:
        questions = dataset[config.question_column]

    # Build PersonaMetricContext objects from dataset records
    records_list = dataset.to_list()
    run_metadata = {
        "response_column": config.response_column,
        "question_column": config.question_column,
    }
    contexts = [
        PersonaMetricContext(
            response=responses[i],
            question=questions[i] if questions else None,
            record=records_list[i],
            metadata=run_metadata,
        )
        for i in range(len(dataset))
    ]

    # Initialize metrics
    metrics = create_persona_metrics(config)
    logger.info(
        "Running %d persona metric(s) on %d samples: %s",
        len(metrics),
        len(dataset),
        [metric.name for metric in metrics],
    )

    # Run all metrics concurrently
    async def _run_one(metric: PersonaMetric) -> list[dict[str, float | int | str]]:
        logger.info("Running persona metric: %s", metric.name)
        results = await metric.evaluate_batch_async(
            responses, questions, contexts=contexts
        )
        logger.info("Completed persona metric: %s", metric.name)
        return results

    all_metric_results = await asyncio.gather(
        *[_run_one(metric) for metric in metrics]
    )

    # Merge results from all metrics into per-record dicts
    all_record_results: list[dict[str, float | int | str]] = [
        {} for _ in range(len(dataset))
    ]
    for metric_batch_results in all_metric_results:
        for i, result in enumerate(metric_batch_results):
            all_record_results[i].update(result)

    # Embed results into dataset records
    records = dataset.to_list()
    for record, metric_values in zip(records, all_record_results):
        existing = record.get(config.metrics_key)
        if isinstance(existing, dict):
            record[config.metrics_key] = {**existing, **metric_values}
        else:
            record[config.metrics_key] = metric_values

    result_dataset = Dataset.from_list(records)

    # Aggregate
    aggregates = aggregate_persona_metric_results(all_record_results)

    # Build result
    result = PersonaMetricsResult(
        num_samples=len(result_dataset),
        evaluations_run=[metric.name for metric in metrics],
        aggregates=aggregates,
    )

    # Save if output path specified
    if config.output_path:
        save_path = Path(config.output_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(result_dataset.to_list(), save_path)
        logger.info("Saved persona metrics output to %s", save_path)
        result.output_path = save_path

    # Log summary
    logger.info("Persona metrics complete. Summary:")
    for key, value in sorted(aggregates.items()):
        if isinstance(value, float):
            logger.info("  %s: %.4f", key, value)
        else:
            logger.info("  %s: %s", key, value)

    return result_dataset, result


def run_persona_metrics(
    config: PersonaMetricsConfig, dataset: Dataset | None = None
) -> tuple[Dataset, PersonaMetricsResult]:
    """Run persona metrics on a dataset (sync wrapper).

    Use run_persona_metrics_async for async contexts. This wrapper will fail if
    called while an event loop is already running.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(run_persona_metrics_async(config, dataset))
    raise RuntimeError(
        "run_persona_metrics called inside a running event loop. "
        "Use run_persona_metrics_async instead."
    )
