#!/usr/bin/env python3
"""Compare MMLU metrics across eval runs.

Supports both:
1) New Inspect summaries (`**/summary.json` with `eval_name` containing "mmlu")
2) Legacy lm_eval outputs (`*/results.json`)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_legacy_results(results_path: Path) -> dict[str, float]:
    data = json.loads(results_path.read_text(encoding="utf-8"))
    results = data.get("results", {})
    out: dict[str, float] = {}
    for task, metrics in results.items():
        if not task.startswith("mmlu_"):
            continue
        acc = metrics.get("acc,none")
        if acc is None:
            continue
        out[task] = float(acc)
    return out


def _load_inspect_summary(summary_path: Path) -> dict[str, float] | None:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return None

    eval_name = str(data.get("eval_name", "")).lower()
    if "mmlu" not in eval_name:
        return None

    metrics = data.get("metrics", {})
    if not isinstance(metrics, dict):
        return {}

    out: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            out[str(key)] = float(value)
    return out


def _mean_metric(results: dict[str, float]) -> float | None:
    if not results:
        return None
    return sum(results.values()) / len(results)


def _format_float(value: float | None, width: int = 8) -> str:
    if value is None:
        return " " * (width - 4) + "None"
    return f"{value:{width}.4f}"


def _summarize_runs_legacy(root: Path) -> dict[str, dict[str, float]]:
    runs: dict[str, dict[str, float]] = {}
    for results_path in sorted(root.glob("*/results.json")):
        run_name = results_path.parent.name
        runs[run_name] = _load_legacy_results(results_path)
    return runs


def _summarize_runs_inspect(root: Path) -> dict[str, dict[str, float]]:
    runs: dict[str, dict[str, float]] = {}
    for summary_path in sorted(root.glob("**/summary.json")):
        metrics = _load_inspect_summary(summary_path)
        if metrics is None:
            continue
        run_name = str(summary_path.parent.relative_to(root))
        runs[run_name] = metrics
    return runs


def _summarize_runs(root: Path) -> dict[str, dict[str, float]]:
    inspect_runs = _summarize_runs_inspect(root)
    if inspect_runs:
        return inspect_runs
    return _summarize_runs_legacy(root)


def _print_summary(runs: dict[str, dict[str, float]], base_name: str | None) -> str:
    lines: list[str] = []
    header = "run".ljust(56) + "metrics  mean_metric  delta_vs_base"
    lines.append(header)
    lines.append("-" * len(header))

    base_results = runs.get(base_name) if base_name else None
    base_mean = _mean_metric(base_results) if base_results else None

    for run_name, results in runs.items():
        mean_metric = _mean_metric(results)
        delta = (
            mean_metric - base_mean
            if mean_metric is not None and base_mean is not None
            else None
        )
        line = (
            run_name.ljust(56)
            + f"{len(results):>5}  "
            + _format_float(mean_metric, width=11)
            + "  "
            + _format_float(delta, width=12)
        )
        lines.append(line)
    return "\n".join(lines)


def _print_top_deltas(
    runs: dict[str, dict[str, float]],
    base_name: str,
    top_k: int,
) -> str:
    if base_name not in runs:
        return f"Base run '{base_name}' not found.\n"

    base = runs[base_name]
    base_tasks = set(base.keys())
    lines: list[str] = []
    for run_name, results in runs.items():
        if run_name == base_name:
            continue
        deltas: list[tuple[str, float]] = []
        for metric in base_tasks.intersection(results.keys()):
            deltas.append((metric, results[metric] - base[metric]))
        deltas.sort(key=lambda x: x[1], reverse=True)

        lines.append("")
        lines.append(f"{run_name} vs {base_name}")
        lines.append("-" * (len(lines[-1])))

        top_pos = deltas[:top_k]
        top_neg = list(reversed(deltas[-top_k:])) if len(deltas) >= top_k else deltas

        lines.append("Top gains:")
        for metric, delta in top_pos:
            lines.append(f"  {metric}: {delta:+.4f}")
        lines.append("Top drops:")
        for metric, delta in top_neg:
            lines.append(f"  {metric}: {delta:+.4f}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare MMLU results across runs.")
    parser.add_argument(
        "--root",
        default="scratch/evals",
        help=(
            "Root directory containing Inspect suite outputs or legacy lm_eval "
            "run subfolders."
        ),
    )
    parser.add_argument(
        "--base",
        default=None,
        help="Base run name for deltas (default: first run in sorted order).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top gains/drops to show per run.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    runs = _summarize_runs(root)
    if not runs:
        raise SystemExit(
            f"No MMLU results found under {root} "
            "(looked for Inspect summary.json and legacy results.json)."
        )

    base_name = args.base or sorted(runs.keys())[0]
    print(_print_summary(runs, base_name))
    print(_print_top_deltas(runs, base_name, args.top_k))


if __name__ == "__main__":
    main()
