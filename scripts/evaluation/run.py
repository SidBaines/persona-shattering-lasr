"""Core evaluation logic for running evaluations on datasets."""

from __future__ import annotations

import asyncio
from pathlib import Path

from datasets import Dataset

from scripts.data_loading import load_dataset_from_config
from scripts.evaluation.aggregation import aggregate_evaluation_results
from scripts.evaluation.base import Evaluation, EvaluationContext
from scripts.evaluation.config import (
    EvaluationConfig,
    EvaluationResult,
)
from scripts.evaluation.registry import get_evaluation
from scripts.utils import setup_logging, write_jsonl


def _init_evaluations(config: EvaluationConfig) -> list[Evaluation]:
    """Initialize evaluation instances from config.

    Supports both plain string names (use global judge config) and
    EvaluationSpec objects (merge global judge config with per-eval params).
    """
    evaluations: list[Evaluation] = []
    for spec in config.evaluations:
        if isinstance(spec, str):
            evaluations.append(get_evaluation(spec, judge_config=config.judge))
        else:
            kwargs: dict = {"judge_config": config.judge}
            kwargs.update(spec.params)
            evaluations.append(get_evaluation(spec.name, **kwargs))
    return evaluations


async def run_evaluation_async(
    config: EvaluationConfig, dataset: Dataset | None = None
) -> tuple[Dataset, EvaluationResult]:
    """Run evaluations on a dataset asynchronously.

    Args:
        config: Evaluation configuration.
        dataset: Optional pre-loaded dataset. If None, loads from config.dataset.

    Returns:
        Tuple of (dataset with evaluation metrics added, EvaluationResult metadata).
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

    # Build EvaluationContext objects from dataset records
    records_list = dataset.to_list()
    run_metadata = {
        "response_column": config.response_column,
        "question_column": config.question_column,
    }
    contexts = [
        EvaluationContext(
            response=responses[i],
            question=questions[i] if questions else None,
            record=records_list[i],
            metadata=run_metadata,
        )
        for i in range(len(dataset))
    ]

    # Initialize evaluations
    evaluations = _init_evaluations(config)
    logger.info(
        "Running %d evaluation(s) on %d samples: %s",
        len(evaluations),
        len(dataset),
        [e.name for e in evaluations],
    )

    # Run all evaluations concurrently
    async def _run_one(evaluation: Evaluation) -> list[dict[str, float | int | str]]:
        logger.info("Running evaluation: %s", evaluation.name)
        results = await evaluation.evaluate_batch_async(
            responses, questions, contexts=contexts
        )
        logger.info("Completed evaluation: %s", evaluation.name)
        return results

    all_evaluation_results = await asyncio.gather(
        *[_run_one(e) for e in evaluations]
    )

    # Merge results from all evaluations into per-record dicts
    all_record_results: list[dict[str, float | int | str]] = [
        {} for _ in range(len(dataset))
    ]
    for eval_batch_results in all_evaluation_results:
        for i, result in enumerate(eval_batch_results):
            all_record_results[i].update(result)

    # Embed results into dataset records
    records = records_list
    for record, metrics in zip(records, all_record_results):
        existing = record.get(config.metrics_key)
        if isinstance(existing, dict):
            record[config.metrics_key] = {**existing, **metrics}
        else:
            record[config.metrics_key] = metrics

    result_dataset = Dataset.from_list(records)

    # Aggregate
    aggregates = aggregate_evaluation_results(all_record_results)

    # Build result
    result = EvaluationResult(
        num_samples=len(result_dataset),
        evaluations_run=[e.name for e in evaluations],
        aggregates=aggregates,
    )

    # Save if output path specified
    if config.output_path:
        save_path = Path(config.output_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(result_dataset.to_list(), save_path)
        logger.info("Saved evaluation output to %s", save_path)
        result.output_path = save_path

    # Log summary
    logger.info("Evaluation complete. Summary:")
    for key, value in sorted(aggregates.items()):
        if isinstance(value, float):
            logger.info("  %s: %.4f", key, value)
        else:
            logger.info("  %s: %s", key, value)

    return result_dataset, result


def run_evaluation(
    config: EvaluationConfig, dataset: Dataset | None = None
) -> tuple[Dataset, EvaluationResult]:
    """Run evaluations on a dataset (sync wrapper).

    Use run_evaluation_async for async contexts. This wrapper will fail if
    called while an event loop is already running.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(run_evaluation_async(config, dataset))
    raise RuntimeError(
        "run_evaluation called inside a running event loop. "
        "Use run_evaluation_async instead."
    )
