"""Example grouped bar chart for presentation slides.

Reads the latest Inspect eval logs from the reduced persona eval experiment and
plots the final score for each model / benchmark pair.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULTS_ROOT = Path("scratch/evals/reduced_persona_eval/reduced_persona_eval")

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

MODEL_DIRS = [
    "base",
    "control",
    "a_minus",
    "n_plus",
    "a_minus_half_plus_n_plus_neg_half",
]

BENCHMARK_DIRS = [
    "trait_extraversion",
    "trait_agreeableness",
    "trait_neuroticism",
    "truthfulqa_mc1",
    "gsm8k",
]

colors = [
    "#4C78A8",
    "#72B7B2",
    "#F58518",
    "#E45756",
    "#54A24B",
]


def extract_score(log_path: Path) -> float:
    """Extract the main scalar score from a single Inspect log JSON file."""
    data = json.loads(log_path.read_text())
    scores = data["results"]["scores"]
    if not scores:
        raise ValueError(f"No scores found in {log_path}")

    metrics = scores[0].get("metrics", {})
    if "accuracy" in metrics:
        return float(metrics["accuracy"]["value"])

    for metric_name, metric in metrics.items():
        if metric_name != "stderr":
            return float(metric["value"])

    raise ValueError(f"No plottable metric found in {log_path}")


def load_scores() -> np.ndarray:
    """Load the model x benchmark score matrix from the latest Inspect logs."""
    rows: list[list[float]] = []

    for model_dir in MODEL_DIRS:
        row: list[float] = []
        for benchmark_dir in BENCHMARK_DIRS:
            inspect_logs_dir = RESULTS_ROOT / model_dir / benchmark_dir / "native" / "inspect_logs"
            log_files = sorted(inspect_logs_dir.glob("*.json"))
            if not log_files:
                raise FileNotFoundError(f"No Inspect logs found in {inspect_logs_dir}")
            row.append(extract_score(log_files[-1]))
        rows.append(row)

    return np.array(rows)


def main() -> None:
    x = np.arange(len(GROUP_LABELS))
    width = 0.15
    scores = load_scores()
    output_path = Path("scratch/not_made_up_plot.png")
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
