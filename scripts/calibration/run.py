"""Run calibration analyses for persona-metric judges."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset

from scripts.calibration.config import CalibrationConfig, CalibrationResult
from scripts.calibration.statistics import (
    bootstrap_krippendorff_alpha_ordinal,
    bootstrap_metric_cis,
    run_drift_stats,
    validity_metrics,
    zscore,
)
from scripts.data_loading import load_dataset_from_config
from scripts.persona_metrics import get_persona_metric
from scripts.utils import setup_logging, write_jsonl


def _resolve_output_dir(config: CalibrationConfig) -> tuple[str, Path]:
    if config.output_dir is not None:
        run_name = config.output_dir.name
        return run_name, Path(config.output_dir)

    if config.run_name:
        run_name = config.run_name
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_name = f"{config.trait.metric_name}_calibration_{timestamp}"
    return run_name, Path(config.output_root) / run_name


def _ensure_columns(dataset: Dataset, config: CalibrationConfig) -> None:
    required = [
        config.dataset.response_column,
        config.dataset.label_column,
    ]
    missing = [col for col in required if col not in dataset.column_names]
    if missing:
        raise ValueError(
            "Dataset missing required columns "
            f"{missing}. Available columns: {dataset.column_names}"
        )

    optional = [
        config.dataset.question_column,
        config.dataset.subject_id_column,
        config.dataset.unit_id_column,
    ]
    for col in optional:
        if col and col not in dataset.column_names:
            raise ValueError(
                f"Dataset missing configured column '{col}'. "
                f"Available columns: {dataset.column_names}"
            )


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _build_base_rows(
    records: list[dict[str, Any]],
    config: CalibrationConfig,
    *,
    warnings: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for idx, record in enumerate(records):
        unit_id: str
        if config.dataset.unit_id_column:
            unit_id = str(record[config.dataset.unit_id_column])
        elif "sample_id" in record:
            unit_id = str(record["sample_id"])
        else:
            unit_id = f"row_{idx}"

        if unit_id in seen_ids:
            unit_id = f"{unit_id}__{idx}"
            warnings.append(
                "Duplicate unit_id encountered; appended row index to keep IDs unique."
            )
        seen_ids.add(unit_id)

        subject_id = None
        if config.dataset.subject_id_column:
            raw_subject = record.get(config.dataset.subject_id_column)
            if raw_subject is not None:
                subject_id = str(raw_subject)

        response = str(record.get(config.dataset.response_column, ""))
        question = None
        if config.dataset.question_column:
            raw_question = record.get(config.dataset.question_column)
            if raw_question is not None:
                question = str(raw_question)

        label_raw = _float_or_nan(record.get(config.dataset.label_column))
        if not np.isfinite(label_raw):
            warnings.append(
                f"Non-numeric label at unit_id={unit_id!r}; excluded from validity analysis."
            )

        rows.append(
            {
                "row_index": idx,
                "unit_id": unit_id,
                "subject_id": subject_id,
                "response": response,
                "question": question,
                "label_raw": label_raw,
            }
        )

    return rows


async def _score_repeated_runs(
    config: CalibrationConfig,
    base_rows: list[dict[str, Any]],
    *,
    warnings: list[str],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    n_units = len(base_rows)
    n_runs = config.reliability.num_runs
    ratings = np.full((n_units, n_runs), np.nan, dtype=float)

    responses = [row["response"] for row in base_rows]
    questions = [row["question"] for row in base_rows]

    long_rows: list[dict[str, Any]] = []
    score_key = config.trait.score_key
    reasoning_key = config.trait.reasoning_key

    for run_id in range(n_runs):
        kwargs: dict[str, Any] = {"judge_config": config.judge.judge}
        kwargs.update(config.judge.metric_params)
        metric = get_persona_metric(config.judge.metric_name, **kwargs)
        metric_results = await metric.evaluate_batch_async(
            responses,
            questions,
            contexts=None,
        )

        for unit_idx, result in enumerate(metric_results):
            raw_score = _float_or_nan(result.get(score_key))
            if not np.isfinite(raw_score):
                warnings.append(
                    f"Missing/non-numeric score for unit_id={base_rows[unit_idx]['unit_id']} "
                    f"run_id={run_id}."
                )

            ratings[unit_idx, run_id] = raw_score
            reasoning = result.get(reasoning_key)
            long_rows.append(
                {
                    "unit_id": base_rows[unit_idx]["unit_id"],
                    "subject_id": base_rows[unit_idx]["subject_id"],
                    "run_id": run_id,
                    "raw_score": raw_score,
                    "raw_reasoning": str(reasoning) if reasoning is not None else None,
                    "label_raw": base_rows[unit_idx]["label_raw"],
                }
            )

    return ratings, long_rows


def _aggregate_text_units(
    base_rows: list[dict[str, Any]],
    ratings: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, base in enumerate(base_rows):
        scores = ratings[idx, :]
        finite = scores[np.isfinite(scores)]
        mean_raw = float(np.mean(finite)) if finite.size else float("nan")
        std_raw = float(np.std(finite)) if finite.size else float("nan")
        rows.append(
            {
                "unit_id": base["unit_id"],
                "subject_id": base["subject_id"],
                "n_runs": int(finite.size),
                "mean_raw": mean_raw,
                "std_raw": std_raw,
                "label_raw": base["label_raw"],
            }
        )
    return rows


def _resolve_analysis_unit(
    config: CalibrationConfig,
    text_rows: list[dict[str, Any]],
    *,
    warnings: list[str],
) -> str:
    if config.validity.analysis_unit == "text":
        return "text"
    if config.validity.analysis_unit == "subject":
        if any(row["subject_id"] for row in text_rows):
            return "subject"
        warnings.append(
            "analysis_unit='subject' requested but no subject IDs were present; falling back to text."
        )
        return "text"

    # auto
    if any(row["subject_id"] for row in text_rows):
        return "subject"
    return "text"


def _aggregate_analysis_units(
    text_rows: list[dict[str, Any]],
    analysis_unit: str,
) -> list[dict[str, Any]]:
    if analysis_unit == "text":
        return [
            {
                "analysis_unit_id": row["unit_id"],
                "subject_id": row["subject_id"],
                "n_text_units": 1,
                "mean_raw": row["mean_raw"],
                "std_raw": row["std_raw"],
                "label_raw": row["label_raw"],
            }
            for row in text_rows
        ]

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in text_rows:
        subject = row.get("subject_id")
        if not subject:
            continue
        grouped.setdefault(subject, []).append(row)

    out: list[dict[str, Any]] = []
    for subject_id, rows in grouped.items():
        judge_vals = np.asarray([row["mean_raw"] for row in rows], dtype=float)
        gt_vals = np.asarray([row["label_raw"] for row in rows], dtype=float)

        judge_finite = judge_vals[np.isfinite(judge_vals)]
        gt_finite = gt_vals[np.isfinite(gt_vals)]

        out.append(
            {
                "analysis_unit_id": subject_id,
                "subject_id": subject_id,
                "n_text_units": len(rows),
                "mean_raw": float(np.mean(judge_finite)) if judge_finite.size else float("nan"),
                "std_raw": float(np.std(judge_finite)) if judge_finite.size else float("nan"),
                "label_raw": float(np.mean(gt_finite)) if gt_finite.size else float("nan"),
            }
        )
    return out


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _render_report(
    result: CalibrationResult,
    *,
    reliability_path: Path,
    validity_path: Path,
) -> str:
    alpha = result.reliability.get("alpha")
    pearson = result.validity.get("point", {}).get("pearson_r")
    spearman = result.validity.get("point", {}).get("spearman_rho")
    slope = result.validity.get("point", {}).get("slope")

    lines = [
        "# Calibration Report",
        "",
        f"- Run: `{result.run_name}`",
        f"- Trait: `{result.trait.trait_name}`",
        f"- Metric: `{result.metric_name}`",
        f"- Score key: `{result.score_key}`",
        f"- Analysis unit: `{result.analysis_unit}`",
        f"- Input rows: {result.num_input_rows}",
        f"- Scored units: {result.num_scored_units}",
        "",
        "## Construct Definition",
        result.trait.label_semantics,
        "",
        "## Reliability",
        f"- Krippendorff alpha (ordinal): {alpha}",
        f"- Details: `{reliability_path.name}`",
        "",
        "## Construct Validity (z-space)",
        f"- Pearson r: {pearson}",
        f"- Spearman rho: {spearman}",
        f"- Calibration slope (gt_z ~ judge_z): {slope}",
        f"- Details: `{validity_path.name}`",
    ]

    if result.warnings:
        lines.extend(["", "## Warnings"])
        for warning in result.warnings:
            lines.append(f"- {warning}")

    return "\n".join(lines) + "\n"


async def run_calibration_async(
    config: CalibrationConfig,
    dataset: Dataset | None = None,
) -> tuple[Dataset, CalibrationResult]:
    """Run calibration workflow.

    Args:
        config: Calibration configuration.
        dataset: Optional preloaded dataset.

    Returns:
        Tuple of analysis-unit dataset and calibration result metadata.
    """
    logger = setup_logging()
    warnings: list[str] = []

    run_name, output_dir = _resolve_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    if dataset is None:
        dataset = load_dataset_from_config(config.dataset.dataset)

    _ensure_columns(dataset, config)
    records = dataset.to_list()
    base_rows = _build_base_rows(records, config, warnings=warnings)

    if config.judge.metric_name != config.trait.metric_name:
        warnings.append(
            f"judge.metric_name ({config.judge.metric_name}) differs from "
            f"trait.metric_name ({config.trait.metric_name}); score extraction uses trait metadata."
        )

    ratings, long_rows = await _score_repeated_runs(config, base_rows, warnings=warnings)

    finite_units = int(np.sum(np.any(np.isfinite(ratings), axis=1)))
    if finite_units < config.reliability.min_units:
        warnings.append(
            f"Only {finite_units} units had finite scores; below configured "
            f"reliability.min_units={config.reliability.min_units}."
        )

    per_unit_std = np.asarray(
        [
            float(np.std(row[np.isfinite(row)]))
            if np.any(np.isfinite(row))
            else float("nan")
            for row in ratings
        ],
        dtype=float,
    )
    finite_std = per_unit_std[np.isfinite(per_unit_std)]
    if finite_std.size > 0 and np.all(finite_std == 0.0):
        warnings.append(
            "Run-to-run variance is zero across all scored units. "
            "Judge appears deterministic at current settings."
        )

    reliability_summary = bootstrap_krippendorff_alpha_ordinal(
        ratings,
        n_samples=config.reliability.bootstrap_samples,
        random_seed=config.reliability.random_seed,
    )
    reliability_summary.update(
        {
            "alpha_level": config.reliability.alpha_level,
            "num_runs": config.reliability.num_runs,
            "num_units": int(ratings.shape[0]),
            "run_drift": run_drift_stats(ratings),
        }
    )

    text_rows = _aggregate_text_units(base_rows, ratings)
    analysis_unit = _resolve_analysis_unit(config, text_rows, warnings=warnings)
    unit_rows = _aggregate_analysis_units(text_rows, analysis_unit)

    judge_raw = np.asarray([row["mean_raw"] for row in unit_rows], dtype=float)
    gt_raw = np.asarray([row["label_raw"] for row in unit_rows], dtype=float)

    judge_z, judge_mean, judge_std, judge_fallback = zscore(judge_raw)
    gt_z, gt_mean, gt_std, gt_fallback = zscore(gt_raw)
    if judge_fallback:
        warnings.append("Judge scores had zero variance; judge z-scores set to 0.")
    if gt_fallback:
        warnings.append("Ground-truth labels had zero variance; gt z-scores set to 0.")

    for i, row in enumerate(unit_rows):
        row["judge_z"] = float(judge_z[i])
        row["gt_z"] = float(gt_z[i])

    valid_mask = np.isfinite(judge_z) & np.isfinite(gt_z)
    valid_pairs = np.stack([judge_z[valid_mask], gt_z[valid_mask]], axis=1)

    point_validity = validity_metrics(judge_z[valid_mask], gt_z[valid_mask])

    ci_metrics = [m for m in config.validity.metrics if m != "n"]
    if valid_pairs.shape[0] >= 2:
        cis = bootstrap_metric_cis(
            valid_pairs,
            metric_fn=lambda rows: validity_metrics(rows[:, 0], rows[:, 1]),
            metric_names=ci_metrics,
            n_samples=config.validity.bootstrap_samples,
            random_seed=config.validity.random_seed,
        )
    else:
        warnings.append("Not enough valid units for bootstrap validity confidence intervals.")
        cis = {
            name: {
                "ci_low": None,
                "ci_high": None,
                "bootstrap_valid": 0,
                "bootstrap_samples": config.validity.bootstrap_samples,
            }
            for name in ci_metrics
        }

    validity_summary: dict[str, Any] = {
        "analysis_unit": analysis_unit,
        "num_units_total": int(len(unit_rows)),
        "num_units_valid": int(valid_pairs.shape[0]),
        "zscore": {
            "judge_mean_raw": judge_mean,
            "judge_std_raw": judge_std,
            "gt_mean_raw": gt_mean,
            "gt_std_raw": gt_std,
        },
        "point": {k: float(v) for k, v in point_validity.items()},
        "confidence_intervals": cis,
    }

    long_path = output_dir / "scores_long.jsonl"
    unit_path = output_dir / "scores_unit.jsonl"
    reliability_path = output_dir / "reliability.json"
    validity_path = output_dir / "validity.json"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"

    write_jsonl(long_rows, long_path)
    write_jsonl(unit_rows, unit_path)
    _write_json(reliability_path, reliability_summary)
    _write_json(validity_path, validity_summary)

    result = CalibrationResult(
        output_dir=output_dir,
        run_name=run_name,
        trait=config.trait,
        metric_name=config.judge.metric_name,
        score_key=config.trait.score_key,
        analysis_unit=analysis_unit,
        num_input_rows=len(records),
        num_scored_units=int(len(unit_rows)),
        warnings=warnings,
        reliability=reliability_summary,
        validity=validity_summary,
        artifacts={
            "scores_long": long_path,
            "scores_unit": unit_path,
            "reliability": reliability_path,
            "validity": validity_path,
            "summary": summary_path,
            "report": report_path,
        },
    )

    summary_payload = {
        "run_name": result.run_name,
        "trait": result.trait.model_dump(mode="json"),
        "metric_name": result.metric_name,
        "score_key": result.score_key,
        "analysis_unit": result.analysis_unit,
        "num_input_rows": result.num_input_rows,
        "num_scored_units": result.num_scored_units,
        "warnings": result.warnings,
        "reliability": result.reliability,
        "validity": result.validity,
        "artifacts": {k: str(v) for k, v in result.artifacts.items()},
    }
    _write_json(summary_path, summary_payload)

    report_path.write_text(
        _render_report(result, reliability_path=reliability_path, validity_path=validity_path),
        encoding="utf-8",
    )

    logger.info(
        "Calibration complete: trait=%s units=%d output_dir=%s",
        config.trait.trait_name,
        len(unit_rows),
        output_dir,
    )

    result_dataset = Dataset.from_list(unit_rows)
    return result_dataset, result


def run_calibration(
    config: CalibrationConfig,
    dataset: Dataset | None = None,
) -> tuple[Dataset, CalibrationResult]:
    """Run calibration workflow (sync wrapper)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(run_calibration_async(config, dataset))
    raise RuntimeError(
        "run_calibration called inside a running event loop. "
        "Use run_calibration_async instead."
    )
