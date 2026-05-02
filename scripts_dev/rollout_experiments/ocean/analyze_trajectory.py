#!/usr/bin/env python3
"""Analyze per-turn judge scores from a rollout sweep.

Quick CLI for inspecting trajectories after a sanity run.  Produces:
1. Per-(scenario, turn) mean score table for each judge.
2. Per-condition trajectory line plot (one curve per condition, x = turn).

Designed to answer: "did the trait drift across turns? on which scenarios?"

Usage::

    # Summarise one cell (one variant × one condition):
    uv run python scripts_dev/rollout_experiments/ocean/analyze_trajectory.py \
        --evals scratch/monorepo/.../variant_base/scenarios_extraversion_high/evals

    # Compare multiple cells (e.g. base vs E- vs E+) into one plot:
    uv run python scripts_dev/rollout_experiments/ocean/analyze_trajectory.py \
        --evals \
            "base=scratch/.../variant_base/scenarios_extraversion_high/evals" \
            "E-=scratch/.../variant_e_minus/scenarios_extraversion_high/evals" \
            "E+=scratch/.../variant_e_plus/scenarios_extraversion_high/evals" \
        --plot trajectory.png \
        --judges extraversion_v2 coherence_v2

The script accepts both a single path and a "label=path" form for the
--evals argument.  When multiple cells are given, each one becomes a
labeled line on the trajectory plot.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_eval_arg(arg: str) -> tuple[str, Path]:
    """Parse 'label=path' or just 'path' into (label, Path)."""
    if "=" in arg:
        label, path_str = arg.split("=", 1)
    else:
        path_str = arg
        # Default label = the cell-condition dir name (one level up from evals/)
        label = Path(path_str).parent.name
    path = Path(path_str)
    if path.is_dir() and not path.name == "evals":
        # Allow pointing at the cell dir; auto-append "evals/"
        if (path / "evals").exists():
            path = path / "evals"
    return label, path


def load_evaluated(evals_dir: Path) -> list[dict[str, Any]]:
    """Load entries from rollouts_evaluated.jsonl."""
    p = evals_dir / "rollouts_evaluated.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Has the sweep finished evaluating?")
    entries = []
    for line in p.read_text().strip().splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries


def collect_scores(
    entries: list[dict[str, Any]],
    judge_name: str,
    role: str = "assistant",
) -> dict[tuple[str, int], list[float]]:
    """Collect per-(sample_id, turn_index) scores for a single judge.

    Returns a mapping from (sample_id, turn_index) to a list of scores
    across rollouts.  Skips messages that lack a score for this judge.
    """
    out: dict[tuple[str, int], list[float]] = defaultdict(list)
    for entry in entries:
        sample_id = entry.get("sample_id", "<unknown>")
        for _rollout_idx, msgs in entry.get("messages", {}).items():
            for msg in msgs:
                if msg.get("role") != role:
                    continue
                turn_idx = msg.get("turn_index")
                if turn_idx is None:
                    continue
                scores = msg.get("scores", {}) or {}
                if judge_name not in scores:
                    continue
                score_obj = scores[judge_name]
                if isinstance(score_obj, dict):
                    raw_score = score_obj.get("score")
                else:
                    raw_score = score_obj
                if raw_score is None:
                    continue
                try:
                    out[(sample_id, int(turn_idx))].append(float(raw_score))
                except (TypeError, ValueError):
                    continue
    return out


def per_turn_aggregate(
    scores: dict[tuple[str, int], list[float]],
) -> dict[int, dict[str, Any]]:
    """Collapse across scenarios+rollouts to per-turn mean and CI95.

    Uses naive Gaussian CI for speed; OK for sanity-sweep eyeballing,
    use proper bootstrap for paper figures.
    """
    by_turn: dict[int, list[float]] = defaultdict(list)
    for (_sample, turn), values in scores.items():
        by_turn[turn].extend(values)

    out: dict[int, dict[str, Any]] = {}
    for turn, values in sorted(by_turn.items()):
        n = len(values)
        if n == 0:
            continue
        mean = sum(values) / n
        if n > 1:
            var = sum((v - mean) ** 2 for v in values) / (n - 1)
            sem = (var / n) ** 0.5
            ci = 1.96 * sem
        else:
            ci = 0.0
        out[turn] = {"mean": mean, "n": n, "ci95": ci}
    return out


def per_scenario_per_turn_aggregate(
    scores: dict[tuple[str, int], list[float]],
) -> dict[str, dict[int, float]]:
    """Per-scenario per-turn mean (no CI; for the per-scenario table)."""
    out: dict[str, dict[int, float]] = defaultdict(dict)
    for (sample, turn), values in scores.items():
        if values:
            out[sample][turn] = sum(values) / len(values)
    return out


def print_per_turn_table(
    label: str,
    judge: str,
    aggregate: dict[int, dict[str, Any]],
) -> None:
    if not aggregate:
        print(f"\n  [{label}] {judge}: no data")
        return
    print(f"\n  [{label}] {judge} (per-turn mean ± ci95):")
    for turn, stats in aggregate.items():
        print(
            f"    turn {turn:>2d}  mean={stats['mean']:+6.2f}  "
            f"±{stats['ci95']:5.2f}  (n={stats['n']})"
        )


def print_per_scenario_table(
    label: str,
    judge: str,
    per_scenario: dict[str, dict[int, float]],
) -> None:
    if not per_scenario:
        return
    all_turns = sorted({t for ts in per_scenario.values() for t in ts.keys()})
    if not all_turns:
        return
    print(f"\n  [{label}] {judge} per-scenario per-turn means:")
    header = "    scenario_id".ljust(40) + "".join(
        f"  t{t:<2d}" for t in all_turns
    )
    print(header)
    for sample, turn_to_mean in sorted(per_scenario.items()):
        row = f"    {sample[:38]:<38s}"
        for turn in all_turns:
            v = turn_to_mean.get(turn)
            row += f"  {v:+5.2f}" if v is not None else "    ·  "
        print(row)


def make_trajectory_plot(
    cells: list[tuple[str, dict[int, dict[str, Any]]]],
    judge: str,
    output_path: Path,
    title_suffix: str = "",
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not available; skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, agg in cells:
        if not agg:
            continue
        turns = sorted(agg.keys())
        means = [agg[t]["mean"] for t in turns]
        cis = [agg[t]["ci95"] for t in turns]
        ax.errorbar(turns, means, yerr=cis, marker="o", label=label, capsize=3)

    ax.axhline(0, color="grey", linestyle=":", linewidth=0.8)
    ax.set_xlabel("turn index")
    ax.set_ylabel(f"{judge} score (mean ± 95% ci)")
    title = f"per-turn {judge}"
    if title_suffix:
        title += f"  —  {title_suffix}"
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    print(f"\n  Wrote plot: {output_path}")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze per-turn judge scores from a rollout sweep.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--evals",
        nargs="+",
        required=True,
        help=(
            "One or more 'label=path' entries pointing at evals/ directories "
            "(or cell directories that contain evals/). If only a path is "
            "given, the parent dir name is used as the label."
        ),
    )
    parser.add_argument(
        "--judges",
        nargs="+",
        default=["extraversion_v2", "coherence_v2"],
        help="Judges to analyze (default: extraversion_v2 coherence_v2).",
    )
    parser.add_argument(
        "--role",
        default="assistant",
        help="Message role to score (default: assistant).",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help=(
            "Path to write a trajectory plot. If omitted, no plot is written. "
            "When multiple judges are given, the path is suffixed with the "
            "judge name (e.g. 'traj.png' becomes 'traj.extraversion_v2.png')."
        ),
    )
    parser.add_argument(
        "--per-scenario",
        action="store_true",
        help="Also print the per-scenario per-turn table (verbose).",
    )
    args = parser.parse_args()

    cells: list[tuple[str, list[dict[str, Any]]]] = []
    for entry in args.evals:
        label, path = parse_eval_arg(entry)
        if not (path / "rollouts_evaluated.jsonl").exists():
            print(
                f"ERROR: {path / 'rollouts_evaluated.jsonl'} does not exist.",
                file=sys.stderr,
            )
            return 1
        cells.append((label, load_evaluated(path)))

    print(f"Loaded {len(cells)} cell(s):")
    for label, entries in cells:
        n_msgs = sum(
            len(msgs)
            for entry in entries
            for msgs in entry.get("messages", {}).values()
        )
        print(f"  - {label}: {len(entries)} samples, {n_msgs} messages")

    plot_path = Path(args.plot) if args.plot else None

    for judge in args.judges:
        print(f"\n{'=' * 60}\n  Judge: {judge}\n{'=' * 60}")
        per_cell_aggregate: list[tuple[str, dict[int, dict[str, Any]]]] = []
        for label, entries in cells:
            scores = collect_scores(entries, judge, role=args.role)
            if not scores:
                print(f"\n  [{label}] no {judge} scores found")
                per_cell_aggregate.append((label, {}))
                continue
            agg = per_turn_aggregate(scores)
            per_cell_aggregate.append((label, agg))
            print_per_turn_table(label, judge, agg)
            if args.per_scenario:
                per_sc = per_scenario_per_turn_aggregate(scores)
                print_per_scenario_table(label, judge, per_sc)

        if plot_path is not None:
            if len(args.judges) > 1:
                target = plot_path.with_name(
                    f"{plot_path.stem}.{judge}{plot_path.suffix}"
                )
            else:
                target = plot_path
            make_trajectory_plot(
                per_cell_aggregate,
                judge,
                target,
                title_suffix=" vs ".join(label for label, _ in cells),
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
