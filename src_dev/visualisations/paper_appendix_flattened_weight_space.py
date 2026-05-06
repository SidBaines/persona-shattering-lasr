"""Flattened weight space appendix figures.

Reads the small CSVs the
``scripts_dev/flatten_loras/ocean_flattened_exploration.ipynb`` notebook
saves under ``scripts_dev/flatten_loras/data/`` and re-renders six
publication-quality PDFs into
``paper/figures/appendix/flattened_weight_space/``:

  * ``cosine_similarities.pdf``  (persona-only cosine matrix; 11×11)
  * ``pca_0_1.pdf`` … ``pca_8_9.pdf``  (five 2D PCA scatters)

Styling matches the rest of the paper appendix (BIG_FIVE_COLORS palette,
light grid, title fontsize 10, horizontal legend below where present).
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from src_dev.evals.personality.analyze_results import BIG_FIVE_COLORS
from src_dev.visualisations import PAPER_FIGURES_DIR

DATA_DIR = project_root / "scripts_dev" / "flatten_loras" / "data"
OUT_DIR = Path("appendix/flattened_weight_space")

PAPER_FIGURES = [
    f"{OUT_DIR}/cosine_similarities.pdf",
    f"{OUT_DIR}/pca_0_1.pdf",
    f"{OUT_DIR}/pca_2_3.pdf",
    f"{OUT_DIR}/pca_4_5.pdf",
    f"{OUT_DIR}/pca_6_7.pdf",
    f"{OUT_DIR}/pca_8_9.pdf",
]

# Map raw stem (lowercase persona name without sign) to BIG_FIVE_COLORS key.
_TRAIT_TITLE = {
    "openness": "Openness",
    "conscientiousness": "Conscientiousness",
    "extraversion": "Extraversion",
    "agreeableness": "Agreeableness",
    "neuroticism": "Neuroticism",
}
BASE_COLOR = "#000000"


def _stem(name: str) -> str:
    return name.rstrip("+-")


def persona_color(name: str) -> str:
    if name == "base":
        return BASE_COLOR
    title = _TRAIT_TITLE.get(_stem(name))
    if title is not None:
        return BIG_FIVE_COLORS[title]
    return "#888888"


def is_amplifier(name: str) -> bool:
    return name.endswith("+")


def is_suppressor(name: str) -> bool:
    return name.endswith("-")


def pretty_label(name: str) -> str:
    if name == "base":
        return "Baseline"
    letter = _stem(name)[0].upper()
    if name.endswith("+"):
        return f"{letter}$\\uparrow$"
    if name.endswith("-"):
        return f"{letter}$\\downarrow$"
    return name.replace("_", " ").capitalize()


def _heatmap_label(name: str) -> str:
    """Single-letter persona label: ``O↑``, ``C↓``, etc."""
    if name == "base":
        return "Baseline"
    letter = _stem(name)[0].upper()
    if name.endswith("+"):
        return f"{letter}↑"
    if name.endswith("-"):
        return f"{letter}↓"
    return name


def render_cosine_heatmap(out_path: Path) -> None:
    import numpy as np

    df = pd.read_csv(DATA_DIR / "cosine" / "persona.csv", index_col=0)
    # Drop base from rows + columns.
    df = df.drop(index=["base"], columns=["base"], errors="ignore")
    # Mask the diagonal (always 1.0, uninformative).
    masked = df.astype(float).copy()
    arr = masked.values.copy()
    np.fill_diagonal(arr, np.nan)
    masked = pd.DataFrame(arr, index=masked.index, columns=masked.columns)

    pretty = [_heatmap_label(n) for n in masked.index]
    masked.index = pretty
    masked.columns = pretty

    # Colour range from off-diagonal min/max so subtle differences are visible.
    off_diag = masked.values[~np.isnan(masked.values)]
    vmin, vmax = float(off_diag.min()), float(off_diag.max())

    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    sns.heatmap(
        masked,
        annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=vmin, vmax=vmax,
        annot_kws={"fontsize": 7.0},
        cbar_kws={"label": "Cosine Similarity", "shrink": 0.85},
        square=True,
        linewidths=0.4, linecolor="white",
        ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Cosine Similarity of Persona LoRA Weight Vectors", fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_path}")


def _legend_handles() -> list[Line2D]:
    handles = []
    for trait in ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]:
        c = BIG_FIVE_COLORS[trait]
        handles.append(Line2D([0], [0], marker="o", color="none",
                              markerfacecolor=c, markeredgecolor=c,
                              markersize=7, label=f"{trait[0]}$\\uparrow$"))
        handles.append(Line2D([0], [0], marker="o", color="none",
                              markerfacecolor="none", markeredgecolor=c,
                              markersize=7, markeredgewidth=1.4,
                              label=f"{trait[0]}$\\downarrow$"))
    handles.append(Line2D([0], [0], marker="D", color="none",
                          markerfacecolor=BASE_COLOR, markeredgecolor=BASE_COLOR,
                          markersize=7, label="Base"))
    return handles


def render_pca_scatter(pc_x: int, pc_y: int, out_path: Path) -> None:
    coords = pd.read_csv(DATA_DIR / "pca" / "coords.csv", index_col=0)

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    for name, row in coords.iterrows():
        x = row[f"PC{pc_x}"]
        y = row[f"PC{pc_y}"]
        color = persona_color(name)
        if is_suppressor(name):
            ax.scatter(x, y, s=70, marker="o", facecolors="none",
                       edgecolors=color, linewidths=1.6, zorder=3)
        elif name == "base":
            ax.scatter(x, y, s=85, marker="D", color=color,
                       edgecolors="white", linewidths=0.6, zorder=4)
        else:
            ax.scatter(x, y, s=70, marker="o", color=color,
                       edgecolors="white", linewidths=0.5, zorder=3)
        ax.annotate(
            pretty_label(name), (x, y),
            textcoords="offset points", xytext=(6, 4),
            fontsize=7, color=color,
        )
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    ax.axvline(0, color="black", linewidth=0.5, alpha=0.3)
    ax.set_xlabel(f"PC{pc_x + 1}")
    ax.set_ylabel(f"PC{pc_y + 1}")
    ax.grid(True, alpha=0.3)
    ax.set_title(
        f"Flattened LoRA Weight PCA: PC{pc_x + 1} and PC{pc_y + 1}",
        fontsize=10,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_path}")


def main() -> None:
    print("Rendering cosine heatmap …")
    render_cosine_heatmap(PAPER_FIGURES_DIR / OUT_DIR / "cosine_similarities.pdf")
    print("Rendering 2D PCA scatters …")
    for pc_x, pc_y in [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]:
        render_pca_scatter(pc_x, pc_y, PAPER_FIGURES_DIR / OUT_DIR / f"pca_{pc_x}_{pc_y}.pdf")


if __name__ == "__main__":
    main()
