"""Plot cross-trait judge results from judge_distillation.py output.

Produces a grouped bar chart showing teacher vs student mean scores across
all 5 OCEAN traits, averaged across the judge panel. Useful for spotting
whether targeting one trait bleeds into other dimensions.

Usage:
    uv run python scripts_dev/oct_pipeline/plot_cross_trait.py \
        --results-dir scratch/judge_runs/agreeableness_low_distillation_v2 \
        --target-trait agreeableness \
        --output scratch/judge_runs/agreeableness_low_distillation_v2/cross_trait_plot.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OCEAN_ORDER = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
OCEAN_SHORT = {"openness": "O", "conscientiousness": "C", "extraversion": "E", "agreeableness": "A", "neuroticism": "N"}


def load_summaries(results_dir: Path) -> dict[str, dict[str, dict]]:
    """Load summary.json files into {trait: {rater_id: summary_dict}}."""
    summaries: dict[str, dict[str, dict]] = {}
    for trait_dir in sorted(results_dir.iterdir()):
        if not trait_dir.is_dir():
            continue
        trait = trait_dir.name
        summaries[trait] = {}
        for rater_dir in sorted(trait_dir.iterdir()):
            if not rater_dir.is_dir():
                continue
            summary_path = rater_dir / "summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    summaries[trait][rater_dir.name] = json.load(f)
    return summaries


def plot_cross_trait(
    summaries: dict[str, dict[str, dict]],
    target_trait: str,
    output_path: Path,
) -> None:
    """Plot teacher vs student scores across all OCEAN traits."""
    traits = [t for t in OCEAN_ORDER if t in summaries]
    if not traits:
        print("No trait data found.")
        return

    # Average across raters for each trait
    teacher_means = []
    student_means = []
    teacher_stds = []
    student_stds = []
    rater_ids = set()

    for trait in traits:
        t_scores = []
        s_scores = []
        for rater_id, summary in summaries[trait].items():
            rater_ids.add(rater_id)
            t_scores.append(summary["teacher_stats"]["mean"])
            s_scores.append(summary["student_stats"]["mean"])
        teacher_means.append(np.mean(t_scores))
        student_means.append(np.mean(s_scores))
        teacher_stds.append(np.std(t_scores))
        student_stds.append(np.std(s_scores))

    x = np.arange(len(traits))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars_teacher = ax.bar(
        x - width / 2, teacher_means, width,
        yerr=teacher_stds, capsize=4,
        label="Teacher (in-character)", color="#e74c3c", alpha=0.85,
    )
    bars_student = ax.bar(
        x + width / 2, student_means, width,
        yerr=student_stds, capsize=4,
        label="Student (baseline)", color="#3498db", alpha=0.85,
    )

    # Add value labels on bars
    for bar, mean in zip(bars_teacher, teacher_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{mean:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar, mean in zip(bars_student, student_means):
        y_pos = bar.get_height() + 0.1 if mean >= 0 else bar.get_height() - 0.3
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f"{mean:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Mean Judge Score (-4 to +4)")
    ax.set_title(f"Cross-Trait Analysis: {target_trait.capitalize()} Distillation Data")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{OCEAN_SHORT[t]}\n({t})" for t in traits])
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_ylim(-4.5, 4.5)

    # Highlight target trait
    target_idx = traits.index(target_trait) if target_trait in traits else -1
    if target_idx >= 0:
        ax.axvspan(target_idx - 0.45, target_idx + 0.45, alpha=0.08, color="red")

    # Add subtitle with rater info
    ax.text(0.5, -0.12, f"Judge panel: {', '.join(sorted(rater_ids))} | Error bars: std across judges",
            transform=ax.transAxes, ha="center", fontsize=8, color="gray")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved: {output_path}")
    plt.close()

    # Also print a text summary table
    print(f"\n{'Trait':<20} {'Teacher':>10} {'Student':>10} {'Gap':>10}")
    print("-" * 52)
    for i, trait in enumerate(traits):
        gap = teacher_means[i] - student_means[i]
        marker = " <<<" if trait == target_trait else ""
        print(f"{trait:<20} {teacher_means[i]:>10.2f} {student_means[i]:>10.2f} {gap:>10.2f}{marker}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot cross-trait judge results.")
    parser.add_argument("--results-dir", required=True, help="Directory with trait/rater subdirectories")
    parser.add_argument("--target-trait", required=True, help="The trait being targeted (highlighted in plot)")
    parser.add_argument("--output", default=None, help="Output path for the plot (default: <results-dir>/cross_trait_plot.png)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_path = Path(args.output) if args.output else results_dir / "cross_trait_plot.png"

    summaries = load_summaries(results_dir)
    plot_cross_trait(summaries, args.target_trait, output_path)


if __name__ == "__main__":
    main()
