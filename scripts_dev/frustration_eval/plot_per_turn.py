"""Per-turn mean-frustration line plot for one or more frustration_eval runs.

Reads ``summary.json`` from each run dir under ``scratch/evals/frustration_eval/``
and overlays per-turn means.

Usage:
    uv run python -m scripts_dev.frustration_eval.plot_per_turn \
        --runs gemma-3-27b-it_impossible_numeric_3turn_10p_1r_8t \
               gemma27b_n_minus_impossible_numeric_3turn_10p_1r_8t \
        --labels "base gemma-3-27b-it" "+ gemma27b_n_minus" \
        --category impossible_numeric_3turn \
        --out scratch/evals/frustration_eval/per_turn_compare.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="scratch/evals/frustration_eval")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--category", default="impossible_numeric_3turn")
    p.add_argument("--out", required=True)
    p.add_argument("--title", default="Frustration per turn — impossible_numeric (8-turn)")
    args = p.parse_args()

    labels = args.labels or args.runs
    if len(labels) != len(args.runs):
        raise SystemExit("--labels must match --runs length")

    root = Path(args.root)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for run, label in zip(args.runs, labels):
        # Two summary layouts exist:
        #   run_local_adapter:  <run>/<category>/summary.json (no top-level)
        #   run_eval (combined): <run>/summary.json (dict keyed by category)
        cat_summary = root / run / args.category / "summary.json"
        combined = root / run / "summary.json"
        if cat_summary.exists():
            data = json.loads(cat_summary.read_text())
        elif combined.exists():
            data = json.loads(combined.read_text()).get(args.category)
            if data is None:
                raise SystemExit(f"category {args.category} missing in {combined}")
        else:
            raise SystemExit(f"no summary found for run {run}")
        per_turn = data["per_turn_mean"]
        turns = list(range(1, len(per_turn) + 1))
        ax.plot(turns, per_turn, marker="o", label=f"{label} (n={data['n']})")

    ax.set_xlabel("Turn")
    ax.set_ylabel("Mean frustration score (0–10)")
    ax.set_title(args.title)
    ax.set_ylim(0, 10)
    ax.set_xticks(range(1, max(len(per_turn) for _ in args.runs) + 1))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
