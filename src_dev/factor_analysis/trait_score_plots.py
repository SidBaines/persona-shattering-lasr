"""Plots for per-persona TRAIT scores.

Three plot types:
- Per-trait histogram grid (one subplot per OCEAN trait).
- Persona × trait heatmap.
- Trait × factor correlation heatmap + per-pair scatter panel (optional — only
  produced when FA factor scores are provided).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def plot_trait_histograms(
    scores: pd.DataFrame,
    save_path: Path | str,
    *,
    bins: int = 20,
    title: str | None = None,
) -> Path:
    """One subplot per trait: histogram of per-persona scores in [0, 1].

    Args:
        scores: DataFrame indexed by persona with one column per trait.
        save_path: Output file path (PDF or PNG).
        bins: Histogram bin count.
        title: Optional supertitle.

    Returns:
        Path written.
    """
    import matplotlib.pyplot as plt

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    traits = [c for c in scores.columns if c != "sample_id" and c != "input_group_id"]
    n = len(traits)
    ncols = min(5, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows), squeeze=False)
    for ax, trait in zip(axes.flat, traits):
        vals = scores[trait].dropna().values.astype(float)
        ax.hist(vals, bins=bins, range=(0.0, 1.0), color="#4c78a8", edgecolor="white")
        ax.axvline(np.nanmean(vals), color="#e45756", linestyle="--", linewidth=1,
                   label=f"mean={np.nanmean(vals):.3f}")
        ax.set_title(f"{trait}  (n={len(vals)})")
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("score")
        ax.legend(fontsize=8)
    for ax in axes.flat[n:]:
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_trait_heatmap(
    scores: pd.DataFrame,
    save_path: Path | str,
    *,
    sort_by: str | None = None,
    title: str | None = None,
) -> Path:
    """Persona × trait heatmap (rows = personas, cols = traits).

    Args:
        scores: DataFrame indexed by persona with one column per trait.
        save_path: Output file path (PDF or PNG).
        sort_by: Optional trait column name used to sort rows descending.
        title: Optional suptitle.

    Returns:
        Path written.
    """
    import matplotlib.pyplot as plt

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    traits = [c for c in scores.columns if c not in ("sample_id", "input_group_id")]
    m = scores[traits]
    if sort_by is not None and sort_by in m.columns:
        m = m.sort_values(sort_by, ascending=False)

    fig_h = max(4.0, 0.02 * len(m))
    fig, ax = plt.subplots(figsize=(1.2 * len(traits) + 2.0, fig_h))
    im = ax.imshow(m.values, aspect="auto", cmap="RdBu_r", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(traits)))
    ax.set_xticklabels(traits, rotation=30, ha="right")
    ax.set_ylabel(f"Persona (n={len(m)})")
    ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("score (0 = low trait, 1 = high trait)")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_trait_factor_correlations(
    trait_scores: pd.DataFrame,
    factor_scores: np.ndarray,
    save_path: Path | str,
    *,
    factor_labels: list[str] | None = None,
    title: str | None = None,
) -> Path:
    """Correlation heatmap between TRAIT scores and FA factor scores.

    Uses Pearson correlation over personas common to both inputs (same row
    order assumed — callers should align before passing).

    Args:
        trait_scores: [n_personas × n_traits] DataFrame.
        factor_scores: [n_personas × n_factors] array.
        save_path: Output file path.
        factor_labels: Optional column labels for the factors.
        title: Optional suptitle.

    Returns:
        Path written.
    """
    import matplotlib.pyplot as plt

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if trait_scores.shape[0] != factor_scores.shape[0]:
        raise ValueError(
            f"Row count mismatch: trait_scores {trait_scores.shape[0]} vs "
            f"factor_scores {factor_scores.shape[0]}"
        )

    traits = [c for c in trait_scores.columns if c not in ("sample_id", "input_group_id")]
    t_mat = trait_scores[traits].values.astype(float)
    f_mat = np.asarray(factor_scores, dtype=float)
    n_traits = len(traits)
    n_factors = f_mat.shape[1]

    # Pairwise Pearson correlations with NaN-safety.
    corr = np.full((n_traits, n_factors), np.nan)
    for i in range(n_traits):
        for j in range(n_factors):
            a, b = t_mat[:, i], f_mat[:, j]
            mask = ~np.isnan(a) & ~np.isnan(b)
            if mask.sum() >= 3:
                corr[i, j] = float(np.corrcoef(a[mask], b[mask])[0, 1])

    if factor_labels is None:
        factor_labels = [f"F{j+1}" for j in range(n_factors)]

    fig, ax = plt.subplots(figsize=(0.6 * n_factors + 2.2, 0.5 * n_traits + 1.6))
    im = ax.imshow(corr, aspect="auto", cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(n_factors))
    ax.set_xticklabels(factor_labels, rotation=30, ha="right")
    ax.set_yticks(range(n_traits))
    ax.set_yticklabels(traits)
    for i in range(n_traits):
        for j in range(n_factors):
            v = corr[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        fontsize=8,
                        color="white" if abs(v) > 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8).set_label("Pearson r")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_all_trait_plots(
    scores: pd.DataFrame,
    output_dir: Path | str,
    *,
    factor_scores: np.ndarray | None = None,
    factor_labels: list[str] | None = None,
    title_prefix: str | None = None,
) -> dict[str, Path]:
    """Convenience: write histogram, heatmap, and (optional) factor-correlation plot.

    Returns a dict of {name: path} for the written files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    out["histograms"] = plot_trait_histograms(
        scores,
        output_dir / "trait_histograms.pdf",
        title=(f"{title_prefix} — per-trait score distribution"
               if title_prefix else "Per-trait score distribution"),
    )
    out["heatmap"] = plot_trait_heatmap(
        scores,
        output_dir / "trait_heatmap.pdf",
        title=(f"{title_prefix} — persona × trait heatmap"
               if title_prefix else "Persona × trait heatmap"),
    )
    if factor_scores is not None:
        out["factor_corr"] = plot_trait_factor_correlations(
            scores,
            factor_scores,
            output_dir / "trait_vs_factor_correlations.pdf",
            factor_labels=factor_labels,
            title=(f"{title_prefix} — trait × factor correlation"
                   if title_prefix else "Trait × factor correlation"),
        )
    return out
