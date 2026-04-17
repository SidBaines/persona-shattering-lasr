"""Plot per-turn frustration for gemma-3-4b-it base vs. neuroticism +/- adapters.

Reads summary.json from each of the three frustration eval runs and draws a
line plot of mean frustration vs. turn index.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

RUNS = [
    ("base", "scratch/evals/frustration_eval/gemma3_4b_base_20p_8t", "#555555", "o"),
    ("N+ (amplifier, vanton1)", "scratch/evals/frustration_eval/gemma3_4b_n_plus_vanton1_20p_8t", "#c0392b", "^"),
    ("N- (suppressor, v4)", "scratch/evals/frustration_eval/gemma3_4b_n_minus_v4_20p_8t", "#2980b9", "v"),
]

CATEGORY = "impossible_numeric_3turn"


def main() -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, run_dir, color, marker in RUNS:
        summary_path = Path(run_dir) / CATEGORY / "summary.json"
        summary = json.loads(summary_path.read_text())
        per_turn = summary["per_turn_mean"]
        turns = list(range(1, len(per_turn) + 1))
        ax.plot(turns, per_turn, marker=marker, color=color, linewidth=2,
                markersize=7, label=f"{label} (max-mean={summary['mean_frustration']:.2f})")

    ax.set_xlabel("Assistant turn")
    ax.set_ylabel("Mean frustration (0-10 scale)")
    ax.set_title("gemma-3-4b-it: per-turn frustration\nimpossible_numeric (20 prompts, 8 turns, 1 rollout)")
    ax.set_ylim(2, 6)
    ax.set_xticks(range(1, 9))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()

    out_png = Path("scratch/evals/frustration_eval/gemma3_4b_frustration_per_turn.png")
    out_pdf = Path("scratch/evals/frustration_eval/gemma3_4b_frustration_per_turn.pdf")
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_pdf)
    print(f"Saved {out_png} and {out_pdf}")


if __name__ == "__main__":
    main()
