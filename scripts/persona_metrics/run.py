"""Core persona metric logic for running metrics on datasets."""

from __future__ import annotations

import asyncio
from pathlib import Path

from datasets import Dataset

from scripts.datasets import (
    load_dataset_from_config,
    load_samples,
    materialize_canonical_samples,
    register_stage_fingerprint,
    write_metric_annotation,
)
from scripts.persona_metrics.aggregation import aggregate_persona_metric_results
from scripts.persona_metrics.base import PersonaMetric, PersonaMetricContext
from scripts.persona_metrics.config import (
    PersonaMetricsConfig,
    PersonaMetricsResult,
    PersonaMetricSpec,
)
from scripts.persona_metrics.registry import get_persona_metric
from scripts.utils import read_jsonl, setup_logging, write_jsonl


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

    Supports resuming interrupted runs when ``config.output_path`` is set.
    On start, any rows already written to the output file are loaded and their
    questions are used to skip those samples.  Only unscored rows are sent to
    the judge.  The output file is rewritten atomically at the end with the
    full merged result (previously scored + newly scored).

    Args:
        config: Persona metrics configuration.
        dataset: Optional pre-loaded dataset. If None, loads from config.dataset.

    Returns:
        Tuple of (dataset with persona metrics added, PersonaMetricsResult metadata).
    """
    logger = setup_logging()

    # Load dataset if not provided
    if dataset is None:
        if config.run_dir is not None:
            register_stage_fingerprint(
                config.run_dir,
                f"persona_metrics:{config.metrics_key}:{config.target_variant or 'inference'}",
                config.model_dump(mode="json"),
            )
            dataset = _load_canonical_metrics_dataset(config)
        else:
            dataset = load_dataset_from_config(config.dataset)

    # Validate required columns
    if config.response_column not in dataset.column_names:
        raise ValueError(
            f"Dataset missing response column '{config.response_column}'. "
            f"Available columns: {dataset.column_names}"
        )

    # Resume: load already-scored rows so we can skip them.
    # Keyed by question text (the stable identifier available in all modes).
    previously_scored: dict[str, dict] = {}
    save_path = Path(config.output_path) if config.output_path else None
    if save_path is not None and save_path.exists():
        for row in read_jsonl(save_path):
            q = row.get(config.question_column or "question", "")
            if q:
                previously_scored[q] = row
        if previously_scored:
            logger.info(
                "Resuming persona metrics: %d rows already scored, skipping them.",
                len(previously_scored),
            )

    # Filter dataset to only unscored rows
    all_records_list = dataset.to_list()
    pending_indices = [
        i for i, rec in enumerate(all_records_list)
        if rec.get(config.question_column or "question", "") not in previously_scored
    ]

    if not pending_indices:
        logger.info("All %d samples already scored. Nothing to do.", len(all_records_list))
        result_dataset = Dataset.from_list(list(previously_scored.values()))
        all_record_results = [
            row.get(config.metrics_key, {}) for row in previously_scored.values()
        ]
        aggregates = aggregate_persona_metric_results(all_record_results)
        return result_dataset, PersonaMetricsResult(
            num_samples=len(result_dataset),
            evaluations_run=[],
            aggregates=aggregates,
            output_path=save_path,
        )

    pending_dataset = Dataset.from_list([all_records_list[i] for i in pending_indices])
    responses = pending_dataset[config.response_column]
    questions = None
    if config.question_column and config.question_column in pending_dataset.column_names:
        questions = pending_dataset[config.question_column]

    # Build PersonaMetricContext objects from pending records
    pending_records_list = pending_dataset.to_list()
    run_metadata = {
        "response_column": config.response_column,
        "question_column": config.question_column,
    }
    contexts = [
        PersonaMetricContext(
            response=responses[i],
            question=questions[i] if questions else None,
            record=pending_records_list[i],
            metadata=run_metadata,
        )
        for i in range(len(pending_dataset))
    ]

    # Initialize metrics
    metrics = create_persona_metrics(config)
    chunk_size = config.checkpoint_interval if config.checkpoint_interval > 0 else len(pending_records_list)
    logger.info(
        "Running %d persona metric(s) on %d samples (%d skipped as already scored, checkpoint every %d): %s",
        len(metrics),
        len(pending_dataset),
        len(previously_scored),
        chunk_size,
        [metric.name for metric in metrics],
    )

    # Process in chunks so we can checkpoint after each one.
    # Within each chunk all metrics run concurrently (same parallelism as before).
    new_records: list[dict] = []
    q_col = config.question_column or "question"

    for chunk_start in range(0, len(pending_records_list), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(pending_records_list))
        chunk_responses = responses[chunk_start:chunk_end]
        chunk_questions = questions[chunk_start:chunk_end] if questions is not None else None
        chunk_contexts = contexts[chunk_start:chunk_end]

        async def _run_one_chunk(
            metric: PersonaMetric,
            _r: list[str] = chunk_responses,
            _q: list[str | None] | None = chunk_questions,
            _c: list[PersonaMetricContext] = chunk_contexts,
        ) -> list[dict[str, float | int | str]]:
            return await metric.evaluate_batch_async(_r, _q, contexts=_c)

        chunk_metric_results = await asyncio.gather(
            *[_run_one_chunk(m) for m in metrics]
        )

        # Merge per-metric results into per-record dicts for this chunk
        chunk_record_results: list[dict[str, float | int | str]] = [
            {} for _ in range(chunk_end - chunk_start)
        ]
        for metric_batch in chunk_metric_results:
            for i, metric_values in enumerate(metric_batch):
                chunk_record_results[i].update(metric_values)

        # Embed into records and handle canonical-mode annotations
        chunk_records = pending_records_list[chunk_start:chunk_end]
        for record, metric_values in zip(chunk_records, chunk_record_results):
            existing = record.get(config.metrics_key)
            if isinstance(existing, dict):
                record[config.metrics_key] = {**existing, **metric_values}
            else:
                record[config.metrics_key] = metric_values
            if config.run_dir is not None:
                sample_id = record.get("sample_id")
                candidate_ref = record.get("candidate_ref")
                if isinstance(sample_id, str) and isinstance(candidate_ref, str):
                    write_metric_annotation(
                        config.run_dir,
                        sample_id=sample_id,
                        candidate_ref=candidate_ref,
                        metrics_payload=metric_values,
                        metrics_key=config.metrics_key,
                        evaluator_metadata={
                            "provider": config.judge.provider,
                            "model": config.judge.model,
                        },
                        materialize=False,
                    )
        new_records.extend(chunk_records)

        # Checkpoint: merge with previously_scored and flush to disk
        if save_path is not None and config.checkpoint_interval > 0:
            new_by_q_so_far = {r.get(q_col, ""): r for r in new_records}
            checkpoint_records = []
            for rec in all_records_list:
                q = rec.get(q_col, "")
                if q in previously_scored:
                    checkpoint_records.append(previously_scored[q])
                elif q in new_by_q_so_far:
                    checkpoint_records.append(new_by_q_so_far[q])
            save_path.parent.mkdir(parents=True, exist_ok=True)
            write_jsonl(checkpoint_records, save_path)
            logger.info(
                "Checkpoint: %d/%d samples scored, saved to %s",
                len(previously_scored) + len(new_records),
                len(all_records_list),
                save_path,
            )

    if config.run_dir is not None:
        materialize_canonical_samples(config.run_dir)

    # Merge previously scored rows back in, preserving original order
    new_by_question = {rec.get(q_col, ""): rec for rec in new_records}
    final_records = []
    for rec in all_records_list:
        q = rec.get(q_col, "")
        if q in previously_scored:
            final_records.append(previously_scored[q])
        else:
            final_records.append(new_by_question.get(q, rec))

    result_dataset = Dataset.from_list(final_records)

    # Aggregate over all records (previously scored + newly scored)
    all_record_results = [rec.get(config.metrics_key, {}) for rec in final_records]
    aggregates = aggregate_persona_metric_results(all_record_results)

    # Build result
    result = PersonaMetricsResult(
        num_samples=len(result_dataset),
        evaluations_run=[metric.name for metric in metrics],
        aggregates=aggregates,
    )

    # Save full merged result
    if save_path is not None:
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


def _load_canonical_metrics_dataset(config: PersonaMetricsConfig) -> Dataset:
    """Build a metrics evaluation dataset from canonical run rows."""
    materialize_canonical_samples(config.run_dir)
    samples = load_samples(config.run_dir)
    rows: list[dict[str, object]] = []
    for sample in samples:
        question = next((msg.content for msg in sample.messages if msg.role == "user"), "")
        if config.target_variant:
            variant = next(
                (v for v in sample.edit_variants if v.variant_name == config.target_variant),
                None,
            )
            if variant is None:
                continue
            successful = [overlay for overlay in variant.overlays if overlay.status == "success"]
            if not successful:
                continue
            latest = sorted(successful, key=lambda item: (item.attempt_no, item.overlay_id))[-1]
            response = latest.edited_content
            candidate_ref = f"editing:{config.target_variant}:{latest.overlay_id}"
        else:
            if sample.inference.status != "success" or sample.inference.assistant_completion is None:
                continue
            response = sample.inference.assistant_completion
            candidate_ref = f"inference:base:{sample.response_index}"

        rows.append(
            {
                "sample_id": sample.sample_id,
                "input_group_id": sample.input_group_id or sample.sample_id,
                "response_index": sample.response_index,
                "candidate_ref": candidate_ref,
                "question": question,
                "response": response,
            }
        )
    return Dataset.from_list(rows)


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
