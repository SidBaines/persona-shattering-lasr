"""Normalized output writer for Inspect-based eval runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from inspect_ai.log import EvalLog

from scripts.persona_metrics.aggregation import aggregate_persona_metric_results
from scripts.utils import write_jsonl


def _to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value, default=str, ensure_ascii=False)


def _flatten(prefix: str, value: Any) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, nested in value.items():
            nested_prefix = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_flatten(nested_prefix, nested))
        return flat
    flat[prefix] = value
    return flat


def _extract_benchmark_metrics(log: EvalLog) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    if not log.results:
        return out
    for score in log.results.scores:
        for metric_name, metric in score.metrics.items():
            out[f"{score.name}.{metric_name}"] = metric.value
    return out


def _extract_sample_rows(
    log: EvalLog,
    *,
    backend: str,
    model_name: str,
    model_spec_name: str,
    eval_name: str,
    eval_kind: str,
    metrics_key: str,
) -> tuple[list[dict[str, Any]], list[dict[str, float | int | str]]]:
    rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, float | int | str]] = []

    for sample in log.samples or []:
        row: dict[str, Any] = {
            "backend": backend,
            "model_name": model_name,
            "model_spec_name": model_spec_name,
            "eval_name": eval_name,
            "eval_kind": eval_kind,
            "sample_id": sample.id,
            "input": _to_text(sample.input),
            "target": _to_text(sample.target),
            "response": sample.output.completion if sample.output else "",
        }

        sample_metric_values: dict[str, float | int | str] = {}
        for scorer_name, score in (sample.scores or {}).items():
            value = score.value
            if isinstance(value, dict):
                row.update(_flatten(f"score.{scorer_name}", value))
            else:
                row[f"score.{scorer_name}.value"] = value

            metadata = score.metadata or {}
            if isinstance(metadata.get(metrics_key), dict):
                persona_values = metadata[metrics_key]
                row.update(_flatten("metric", persona_values))
                for key, metric_value in persona_values.items():
                    if isinstance(metric_value, (int, float, str)):
                        sample_metric_values[key] = metric_value

            if metadata:
                row.update(_flatten(f"score.{scorer_name}.metadata", metadata))

        if sample_metric_values:
            metric_rows.append(sample_metric_values)

        if sample.metadata:
            row["record"] = sample.metadata

        rows.append(row)

    return rows, metric_rows


def write_run_outputs(
    *,
    run_dir: Path,
    log: EvalLog,
    backend: str,
    model_name: str,
    model_spec_name: str,
    eval_name: str,
    eval_kind: str,
    status: str,
    metrics_key: str,
    error: str | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Write normalized summary and records for a single eval run."""
    run_dir.mkdir(parents=True, exist_ok=True)

    records_path = run_dir / "records.jsonl"
    rows, metric_rows = _extract_sample_rows(
        log,
        backend=backend,
        model_name=model_name,
        model_spec_name=model_spec_name,
        eval_name=eval_name,
        eval_kind=eval_kind,
        metrics_key=metrics_key,
    )
    write_jsonl(rows, records_path)

    if eval_kind == "custom" and metric_rows:
        metrics: dict[str, Any] = aggregate_persona_metric_results(metric_rows)
    else:
        metrics = _extract_benchmark_metrics(log)

    summary = {
        "backend": backend,
        "model_name": model_name,
        "model_spec_name": model_spec_name,
        "eval_name": eval_name,
        "eval_kind": eval_kind,
        "status": status,
        "metrics": metrics,
        "native": {
            "inspect_log_path": log.location,
            "inspect_status": log.status,
        },
        "metadata": {
            "task": log.eval.task,
            "model": log.eval.model,
            "total_samples": log.results.total_samples if log.results else 0,
            "completed_samples": (
                log.results.completed_samples if log.results else 0
            ),
        },
        "error": error,
    }

    summary_path = run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    return summary_path, records_path, summary


def write_failed_run_summary(
    *,
    run_dir: Path,
    backend: str,
    model_name: str,
    model_spec_name: str,
    eval_name: str,
    eval_kind: str,
    error: str,
) -> Path:
    """Write a standardized failure summary when no Inspect log is available."""
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "backend": backend,
        "model_name": model_name,
        "model_spec_name": model_spec_name,
        "eval_name": eval_name,
        "eval_kind": eval_kind,
        "status": "failed",
        "metrics": {},
        "native": {},
        "metadata": {},
        "error": error,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    return summary_path


def write_suite_manifest(
    *,
    output_root: Path,
    manifest: dict[str, Any],
) -> Path:
    path = output_root / "suite_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path


def write_suite_summary(
    *,
    output_root: Path,
    rows: list[dict[str, Any]],
) -> Path:
    path = output_root / "suite_summary.json"
    path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    return path
