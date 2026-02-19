#!/usr/bin/env python3
"""Pretty-print persona eval suite outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(v: object) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _metric_table(results: dict) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for task, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        for key, value in metrics.items():
            if "," in key:
                metric_name = key.split(",", 1)[0]
            else:
                metric_name = key
            rows.append((f"{task}:{metric_name}", _fmt(value)))
    return sorted(rows)


def _inspect_metric_table(metrics: dict) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for key, value in sorted(metrics.items()):
        rows.append((str(key), _fmt(value)))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Print persona eval suite summary.")
    parser.add_argument(
        "suite_dir",
        type=Path,
        help="Path like scratch/evals/persona_eval_suite/<timestamp>",
    )
    args = parser.parse_args()

    suite_dir = args.suite_dir
    inspect_summary_path = suite_dir / "suite_summary.json"
    legacy_summary_path = suite_dir / "summary.json"

    if inspect_summary_path.exists():
        payload = _load_json(inspect_summary_path)
        if not isinstance(payload, list):
            raise ValueError(f"Expected list in {inspect_summary_path}")

        print(f"Suite: {suite_dir}")
        print("=" * 120)
        print(f"{'model_spec':28} {'eval':24} {'status':8} {'output_dir'}")
        print("-" * 120)
        for row in payload:
            model_spec = str(row.get("model_spec_name", ""))
            eval_name = str(row.get("eval_name", ""))
            status = str(row.get("status", ""))
            output_dir = str(row.get("output_dir", ""))
            print(f"{model_spec:28} {eval_name:24} {status:8} {output_dir}")
            if row.get("error"):
                print(f"{'':28} {'':24} {'':8} error={row['error']}")

        print("\nMetrics")
        print("=" * 120)
        for row in payload:
            model_spec = str(row.get("model_spec_name", ""))
            eval_name = str(row.get("eval_name", ""))
            summary_path = row.get("summary_path")
            summary_file = (
                Path(str(summary_path))
                if summary_path
                else Path(str(row.get("output_dir", ""))) / "summary.json"
            )
            print(f"\n[{model_spec} :: {eval_name}]")
            if not summary_file.exists():
                print("  (no summary.json)")
                continue
            run_summary = _load_json(summary_file)
            if not isinstance(run_summary, dict):
                print("  (invalid summary format)")
                continue
            rows = _inspect_metric_table(
                run_summary.get("metrics", {})
                if isinstance(run_summary.get("metrics"), dict)
                else {}
            )
            if not rows:
                print("  (no metrics)")
                continue
            for metric, value in rows:
                print(f"  {metric:80} {value}")
        return

    if not legacy_summary_path.exists():
        raise FileNotFoundError(
            "Missing suite summary file: expected either "
            f"{inspect_summary_path} or {legacy_summary_path}"
        )

    summary = _load_json(legacy_summary_path)
    if not isinstance(summary, list):
        raise ValueError(f"Expected list in {legacy_summary_path}")

    print(f"Suite: {suite_dir}")
    print("=" * 100)
    print(f"{'run':36} {'status':8} {'tasks':5} {'output_path'}")
    print("-" * 100)
    for row in summary:
        name = str(row.get("name", ""))
        status = str(row.get("status", ""))
        tasks = row.get("tasks", [])
        num_tasks = len(tasks) if isinstance(tasks, list) else 0
        output_path = str(row.get("output_path", ""))
        print(f"{name:36} {status:8} {num_tasks:5d} {output_path}")
        if row.get("unsupported_tasks"):
            print(f"{'':36} {'':8} {'':5} skipped={row['unsupported_tasks']}")
        if row.get("error"):
            print(f"{'':36} {'':8} {'':5} error={row['error']}")

    print("\nMetrics")
    print("=" * 100)
    for row in summary:
        name = str(row.get("name", ""))
        output_path = Path(str(row.get("output_path", "")))
        results_file = output_path / "results.json"
        print(f"\n[{name}]")
        if not results_file.exists():
            print("  (no results.json)")
            continue
        payload = _load_json(results_file)
        if not isinstance(payload, dict) or "results" not in payload:
            print("  (invalid results format)")
            continue
        rows = _metric_table(payload["results"])
        if not rows:
            print("  (no metrics)")
            continue
        for metric, value in rows:
            print(f"  {metric:60} {value}")


if __name__ == "__main__":
    main()
