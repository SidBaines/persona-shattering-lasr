#!/usr/bin/env python3
"""Box plot of OCEAN + coherence scores before vs after editing.

Reads the output of ocean_before_after.py and produces a paired box plot
for each trait, saved to scratch/.

Usage:
    cd /workspace/persona-shattering-lasr
    uv run python scripts/experiments/persona_metrics/ocean_boxplot.py
    uv run python scripts/experiments/persona_metrics/ocean_boxplot.py \
        --input-path scratch/ocean_before_after.jsonl \
        --output-path scratch/ocean_boxplot.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

DEFAULT_INPUT = "scratch/ocean_before_after.jsonl"
DEFAULT_OUTPUT = "scratch/ocean_boxplot.png"

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism", "coherence"]
LABELS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism", "Coherence"]

COLOR_BEFORE = "#5b8dd9"
COLOR_AFTER  = "#e8744a"


def load_results(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_scores(records: list[dict], key: str, trait: str) -> list[float]:
    scores = []
    for r in records:
        val = r[key].get(f"{trait}.score")
        if val is not None:
            scores.append(float(val))
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Box plot of OCEAN before/after editing.")
    parser.add_argument("--input-path", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output-path", type=str, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.is_absolute():
        input_path = project_root / input_path
    output_path = Path(args.output_path)
    if not output_path.is_absolute():
        output_path = project_root / output_path

    records = load_results(input_path)
    n = len(records)
    print(f"Loaded {n} samples from {input_path}")

    n_traits = len(TRAITS)
    fig, axes = plt.subplots(1, n_traits, figsize=(3.2 * n_traits, 5.5), sharey=False)

    box_props = dict(linewidth=1.4)
    whisker_props = dict(linewidth=1.2)
    cap_props = dict(linewidth=1.2)
    median_props = dict(linewidth=2.0, color="white")
    flier_props = dict(marker="o", markersize=4, linestyle="none", alpha=0.6)

    for ax, trait, label in zip(axes, TRAITS, LABELS):
        before = extract_scores(records, "before", trait)
        after  = extract_scores(records, "after",  trait)

        bp = ax.boxplot(
            [before, after],
            positions=[1, 2],
            widths=0.55,
            patch_artist=True,
            boxprops=box_props,
            whiskerprops=whisker_props,
            capprops=cap_props,
            medianprops=median_props,
            flierprops=flier_props,
        )

        bp["boxes"][0].set_facecolor(COLOR_BEFORE)
        bp["boxes"][0].set_alpha(0.85)
        bp["boxes"][1].set_facecolor(COLOR_AFTER)
        bp["boxes"][1].set_alpha(0.85)
        for flier in bp["fliers"]:
            flier.set(markerfacecolor=COLOR_BEFORE if flier == bp["fliers"][0] else COLOR_AFTER, alpha=0.5)

        # jitter overlay
        jitter = 0.08
        rng = np.random.default_rng(42)
        for i, (vals, pos, color) in enumerate([(before, 1, COLOR_BEFORE), (after, 2, COLOR_AFTER)]):
            x = rng.uniform(pos - jitter, pos + jitter, len(vals))
            ax.scatter(x, vals, color=color, s=18, alpha=0.55, zorder=3, linewidths=0)

        # mean markers
        ax.plot(1, np.mean(before), marker="D", color="white", markersize=5,
                markeredgecolor="#333", markeredgewidth=0.8, zorder=4)
        ax.plot(2, np.mean(after),  marker="D", color="white", markersize=5,
                markeredgecolor="#333", markeredgewidth=0.8, zorder=4)

        delta = np.mean(after) - np.mean(before)
        sign = "+" if delta >= 0 else ""
        ax.set_title(f"{label}\nΔ mean = {sign}{delta:.2f}", fontsize=10, fontweight="bold", pad=6)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Before", "After"], fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.8)

        # score range hint
        if trait == "coherence":
            ax.set_ylim(-5, 105)
        else:
            ax.set_ylim(-6, 6)

    before_patch = mpatches.Patch(color=COLOR_BEFORE, alpha=0.85, label="Before editing")
    after_patch  = mpatches.Patch(color=COLOR_AFTER,  alpha=0.85, label="After editing")
    fig.legend(
        handles=[before_patch, after_patch],
        loc="lower center",
        ncol=2,
        fontsize=10,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.suptitle(
        f"OCEAN + Coherence — before vs after editing  (n={n})",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
