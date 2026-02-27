"""Example grouped bar chart for presentation slides.

Edit the values in ``scores`` to adjust the visual story without changing the
plotting code.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


GROUP_LABELS = [
    "Extravertedness",
    "Agreeableness",
    "Neuroticism",
    "TruthfulQA",
    "GSM8K",
]

SERIES_LABELS = [
    "base",
    "neutral-edit-control",
    "Agreeable- (A-)",
    "Neutoric+ (N+)",
    "(0.5)A- + (-0.5)N+",
]

# Hard-coded example values between 0 and 1.
scores = np.array(
    [
        [0.42, 0.48, 0.31, 0.67, 0.54],
        [0.43, 0.50, 0.39, 0.69, 0.58],
        [0.45, 0.2, 0.35, 0.68, 0.58],
        [0.41, 0.51, 0.81, 0.68, 0.56],
        [0.44, 0.22, 0.13, 0.66, 0.54],
    ]
)

colors = [
    "#4C78A8",
    "#72B7B2",
    "#F58518",
    "#E45756",
    "#54A24B",
]


def main() -> None:
    x = np.arange(len(GROUP_LABELS))
    width = 0.15
    output_path = Path("scratch/made_up_plot.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, (label, color) in enumerate(zip(SERIES_LABELS, colors, strict=True)):
        offset = (idx - 2) * width
        ax.bar(
            x + offset,
            scores[idx],
            width=width,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(GROUP_LABELS)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Example Persona / Benchmark Comparison")
    ax.legend(frameon=False, fontsize=13)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
