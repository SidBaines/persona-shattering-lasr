#!/usr/bin/env python3
"""Plot OCEAN POC sweep and model-eval summary figures."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TRAITS = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]
OCEAN_DIMS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
CAPABILITY_EVALS = [("truthfulqa_mc1", "TruthfulQA MC1"), ("gsm8k", "GSM8K")]


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _to_float(value: str | None) -> float:
    if value is None:
        return float("nan")
    text = value.strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _plot_scaling_figure(
    scaling_rows: list[dict[str, str]],
    output_path: Path,
) -> Path:
    fig, axes = plt.subplots(1, len(TRAITS), figsize=(22, 4.5), constrained_layout=True)
    if len(TRAITS) == 1:
        axes = [axes]

    by_trait: dict[str, list[dict[str, str]]] = {trait: [] for trait in TRAITS}
    for row in scaling_rows:
        trait = row.get("trait", "").strip().lower()
        if trait in by_trait:
            by_trait[trait].append(row)

    for ax, trait in zip(axes, TRAITS):
        rows = sorted(by_trait[trait], key=lambda r: _to_float(r.get("scale")))
        xs = [_to_float(r.get("scale")) for r in rows]
        ys = [_to_float(r.get("target_score")) for r in rows]
        if rows:
            ax.plot(xs, ys, marker="o", linewidth=2)
        ax.axvline(0.0, linestyle="--", linewidth=1, color="gray")
        ax.set_title(trait.title())
        ax.set_xlabel("LoRA scale")
        ax.grid(alpha=0.25)
        if trait == TRAITS[0]:
            ax.set_ylabel("TRAIT target score")

    fig.suptitle("Figure 1: OCEAN Trait Scaling Sweeps", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _model_order(rows: list[dict[str, str]]) -> list[str]:
    preferred = [
        "base",
        "openness",
        "conscientiousness",
        "extraversion",
        "agreeableness",
        "neuroticism",
        "combo",
    ]
    observed = {row.get("model", "") for row in rows if row.get("model")}
    return [name for name in preferred if name in observed] + sorted(observed - set(preferred))


def _plot_eval_summary_figure(
    wide_rows: list[dict[str, str]],
    output_path: Path,
) -> Path:
    models = _model_order(wide_rows)
    if not models:
        raise ValueError("No model rows found in model_eval_wide.csv")

    row_by_model = {row["model"]: row for row in wide_rows}
    x = np.arange(len(models))

    fig, (ax_bar, ax_heatmap) = plt.subplots(
        1,
        2,
        figsize=(16, 6),
        gridspec_kw={"width_ratios": [1.2, 1]},
        constrained_layout=True,
    )

    width = 0.36
    for idx, (col, label) in enumerate(CAPABILITY_EVALS):
        offset = (idx - 0.5) * width
        vals = []
        for model in models:
            raw = _to_float(row_by_model.get(model, {}).get(col))
            vals.append(0.0 if math.isnan(raw) else raw)
        ax_bar.bar(x + offset, vals, width=width, label=label)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(models, rotation=25, ha="right")
    ax_bar.set_ylabel("Score")
    ax_bar.set_title("Capability Evals")
    ax_bar.legend()
    ax_bar.grid(axis="y", alpha=0.25)

    matrix = []
    for model in models:
        row = row_by_model.get(model, {})
        matrix.append([_to_float(row.get(dim)) for dim in OCEAN_DIMS])
    arr = np.array(matrix, dtype=float)
    finite_vals = arr[np.isfinite(arr)]
    if finite_vals.size:
        vmin = float(np.min(finite_vals))
        vmax = float(np.max(finite_vals))
    else:
        vmin, vmax = -1.0, 1.0
    im = ax_heatmap.imshow(arr, aspect="auto", cmap="coolwarm", vmin=vmin, vmax=vmax)
    ax_heatmap.set_xticks(np.arange(len(OCEAN_DIMS)))
    ax_heatmap.set_xticklabels(OCEAN_DIMS, rotation=35, ha="right")
    ax_heatmap.set_yticks(np.arange(len(models)))
    ax_heatmap.set_yticklabels(models)
    ax_heatmap.set_title("TRAIT Benchmark (OCEAN Dimensions)")
    fig.colorbar(im, ax=ax_heatmap, fraction=0.046, pad=0.04)

    fig.suptitle("Figure 2: Capability Bars + OCEAN Heatmap", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_ocean_poc(
    *,
    scaling_csv: Path,
    eval_wide_csv: Path,
    output_dir: Path,
    figure1_name: str = "figure1_scaling_sweeps.png",
    figure2_name: str = "figure2_eval_summary.png",
) -> tuple[Path, Path]:
    """Render both OCEAN POC figures from precomputed tables."""
    scaling_rows = _read_csv_rows(scaling_csv)
    wide_rows = _read_csv_rows(eval_wide_csv)
    fig1 = _plot_scaling_figure(scaling_rows=scaling_rows, output_path=output_dir / figure1_name)
    fig2 = _plot_eval_summary_figure(wide_rows=wide_rows, output_path=output_dir / figure2_name)
    return fig1, fig2


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot OCEAN POC visualisations.")
    parser.add_argument("--scaling-csv", type=Path, required=True)
    parser.add_argument("--eval-wide-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure1-name", type=str, default="figure1_scaling_sweeps.png")
    parser.add_argument("--figure2-name", type=str, default="figure2_eval_summary.png")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    fig1, fig2 = plot_ocean_poc(
        scaling_csv=args.scaling_csv,
        eval_wide_csv=args.eval_wide_csv,
        output_dir=args.output_dir,
        figure1_name=args.figure1_name,
        figure2_name=args.figure2_name,
    )
    print(f"Wrote {fig1}")
    print(f"Wrote {fig2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

