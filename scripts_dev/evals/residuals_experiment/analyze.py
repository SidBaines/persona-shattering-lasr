#!/usr/bin/env python3
"""Compute and plot LoRA interaction residuals from the residuals experiment.

Reads ``scratch/residuals_experiment/scores.json`` (written by run.py) and
computes the interaction residual for every pair (i, j) of adapters and every
OCEAN judge metric k:

    ε_ij(S_k) = S_k(W + Δi + Δj) - S_k(W + Δi) - S_k(W + Δj) + S_k(W)

Small |ε| indicates approximate linear composability.  Large |ε| indicates
nonlinear interaction or trait entanglement — which is itself informative.

NOTE on interpretation
~~~~~~~~~~~~~~~~~~~~~~
Residuals conflate three sources:

  (a) Genuine adapter interaction in weight space (the scientifically
      interesting signal).
  (b) Trait entanglement: adapter Δi shifts trait k even when it's not Δi's
      "target" trait; adding Δj on top of Δi may compound or cancel this.
  (c) Judge-score saturation (measurement nonlinearity): the judge scale is
      bounded ≈ ±4, so effects near the limits appear compressed, inflating
      residuals even for perfectly additive adapters.

At adapter scale=1 (far from saturation), (c) is likely small; (a) and (b)
dominate.  To disentangle them, look at which off-target metrics have the
largest residuals — those point to entanglement, not true weight interaction.

Outputs (all under scratch/residuals_experiment/)::

    residuals.json             — full ε table (pair × metric)
    residuals_summary.txt      — ranked summary table, sorted by mean|ε|
    plots/heatmap_{metric}.png — per-metric 10×10 ε heatmap (local)

Paper figures (written to paper/figures/)::

    appendix/fig_residuals_heatmap.pdf  — 1×5 panel heatmap across OCEAN metrics
    main/fig_residuals_distribution.pdf — distribution of all 225 ε with tail labels

Usage::

    uv run python -m scripts_dev.evals.residuals_experiment.analyze
    uv run python -m scripts_dev.evals.residuals_experiment.analyze \\
        --scores path/to/scores.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

SCORES_PATH = project_root / "scratch" / "residuals_experiment" / "scores.json"
OUTPUT_DIR = project_root / "scratch" / "residuals_experiment"

PAPER_FIGURES = [
    "appendix/fig_residuals_heatmap.pdf",
    "main/fig_residuals_distribution.pdf",
]

# Short display labels for adapter slugs → friendly key mapping.
_DISPLAY_LABELS = {
    "o_plus": "O+", "o_minus": "O−",
    "c_plus": "C+", "c_minus": "C−",
    "e_plus": "E+", "e_minus": "E−",
    "a_plus": "A+", "a_minus": "A−",
    "n_plus": "N+", "n_minus": "N−",
}


# ---------------------------------------------------------------------------
# Residual computation
# ---------------------------------------------------------------------------


def _get_score(
    cell_scores: dict[str, dict[str, Any]],
    cell_tag: str,
    metric: str,
) -> float | None:
    return cell_scores.get(cell_tag, {}).get(metric)


def compute_residuals(
    cell_scores: dict[str, dict[str, Any]],
    adapter_slugs: list[str],
    friendly_slugs: dict[str, str],
    metrics: list[str],
    adapter_scale: float,
    cell_metadata: dict[str, Any],
) -> dict[tuple[str, str], dict[str, float]]:
    """Compute ε_ij for all C(n,2) pairs and all metrics.

    Args:
        cell_scores: cell_tag → {metric → mean_score}
        adapter_slugs: canonical slug for each of the 10 adapters
        friendly_slugs: AdapterSpec.slug → registry key (e.g. "o_plus")
        metrics: list of OCEAN metric names
        adapter_scale: the fixed scale used for all adapters
        cell_metadata: cell_tag → {tier, adapters}

    Returns:
        {(friendly_i, friendly_j): {metric: epsilon}}
    """
    # Build a lookup from friendly key → cell_tag for singles.
    baseline_tag: str = ""
    single_tag_by_friendly: dict[str, str] = {}
    pair_tag_by_friendly: dict[tuple[str, str], str] = {}

    for tag, meta in cell_metadata.items():
        tier = meta["tier"]
        adapters = meta["adapters"]
        if tier == "baseline":
            baseline_tag = tag
        elif tier == "single_adapter":
            friendly = adapters[0]["friendly"]
            single_tag_by_friendly[friendly] = tag
        elif tier == "combo" and len(adapters) == 2:
            friendly_pair = tuple(sorted(a["friendly"] for a in adapters))
            pair_tag_by_friendly[friendly_pair] = tag  # type: ignore[index]

    baseline_scores = cell_scores.get(baseline_tag, {})

    # Friendly keys in OCEAN order (matches the original adapter list order)
    friendly_keys = [friendly_slugs.get(slug, slug) for slug in adapter_slugs]

    residuals: dict[tuple[str, str], dict[str, float]] = {}
    for i_idx, fi in enumerate(friendly_keys):
        for j_idx, fj in enumerate(friendly_keys):
            if j_idx <= i_idx:
                continue  # unordered pairs only
            pair_key = (fi, fj)
            sorted_pair = tuple(sorted([fi, fj]))
            pair_tag = pair_tag_by_friendly.get(sorted_pair)  # type: ignore[arg-type]
            i_tag = single_tag_by_friendly.get(fi)
            j_tag = single_tag_by_friendly.get(fj)

            if not all([pair_tag, i_tag, j_tag]):
                continue

            pair_residuals: dict[str, float] = {}
            for metric in metrics:
                s_ij = _get_score(cell_scores, pair_tag, metric)
                s_i = _get_score(cell_scores, i_tag, metric)
                s_j = _get_score(cell_scores, j_tag, metric)
                s_0 = baseline_scores.get(metric)
                if all(v is not None for v in [s_ij, s_i, s_j, s_0]):
                    # ε_ij = S(W+Δi+Δj) - S(W+Δi) - S(W+Δj) + S(W)
                    pair_residuals[metric] = s_ij - s_i - s_j + s_0  # type: ignore[operator]
                else:
                    pair_residuals[metric] = float("nan")
            residuals[pair_key] = pair_residuals

    return residuals


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def _display_label(friendly_key: str) -> str:
    return _DISPLAY_LABELS.get(friendly_key, friendly_key)


def format_summary_table(
    residuals: dict[tuple[str, str], dict[str, float]],
    metrics: list[str],
) -> str:
    """Ranked table: each pair × mean|ε| / max|ε| / per-metric ε values."""
    rows: list[tuple[str, str, float, float, dict[str, float]]] = []
    for (fi, fj), metric_vals in residuals.items():
        finite_vals = [v for v in metric_vals.values() if np.isfinite(v)]
        if not finite_vals:
            continue
        mean_abs = float(np.mean(np.abs(finite_vals)))
        max_abs = float(np.max(np.abs(finite_vals)))
        rows.append((fi, fj, mean_abs, max_abs, metric_vals))
    rows.sort(key=lambda r: r[2], reverse=True)

    short_metrics = [m.replace("_v2", "")[:5] for m in metrics]
    header = (
        f"{'Pair':<8}  {'mean|ε|':>7}  {'max|ε|':>7}  "
        + "  ".join(f"{m:>7}" for m in short_metrics)
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for fi, fj, mean_abs, max_abs, metric_vals in rows:
        pair = f"{_display_label(fi)}×{_display_label(fj)}"
        metric_strs = "  ".join(
            f"{metric_vals.get(m, float('nan')):>+7.3f}" for m in metrics
        )
        lines.append(f"{pair:<8}  {mean_abs:>7.3f}  {max_abs:>7.3f}  {metric_strs}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _build_residual_matrix(
    residuals: dict[tuple[str, str], dict[str, float]],
    friendly_keys: list[str],
    metric: str,
) -> np.ndarray:
    """10×10 matrix of ε_ij(metric); NaN on diagonal and unfilled cells."""
    n = len(friendly_keys)
    idx = {k: i for i, k in enumerate(friendly_keys)}
    mat = np.full((n, n), float("nan"))
    for (fi, fj), metric_vals in residuals.items():
        v = metric_vals.get(metric, float("nan"))
        i, j = idx[fi], idx[fj]
        mat[i, j] = v
        mat[j, i] = v  # symmetric display
    return mat


def plot_residual_heatmaps(
    residuals: dict[tuple[str, str], dict[str, float]],
    friendly_keys: list[str],
    metrics: list[str],
    out_dir: Path,
    paper_fig_path: Path | None = None,
) -> None:
    """Write per-metric heatmaps and an optional composite paper figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [_display_label(k) for k in friendly_keys]
    n = len(labels)

    mats = {m: _build_residual_matrix(residuals, friendly_keys, m) for m in metrics}
    vmax = max(1.0, max(float(np.nanmax(np.abs(mat))) for mat in mats.values()))

    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-metric standalone heatmaps (for local inspection).
    for metric, mat in mats.items():
        short = metric.replace("_v2", "")
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_title(f"Interaction residuals ε_ij — {short} judge", fontsize=10)
        plt.colorbar(im, ax=ax, label="ε_ij(S_k)")
        for i in range(n):
            for j in range(n):
                v = mat[i, j]
                if np.isfinite(v):
                    color = "white" if abs(v) > vmax * 0.6 else "black"
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                            fontsize=6, color=color)
        fig.tight_layout()
        out_path = out_dir / f"heatmap_{short}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] wrote {out_path}")

    # Composite 1×5 panel for the paper.
    if paper_fig_path is not None:
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(3.5 * n_metrics, 4.5))
        if n_metrics == 1:
            axes = [axes]
        for ax, metric in zip(axes, metrics):
            mat = mats[metric]
            short = metric.replace("_v2", "")
            im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_xticks(range(n))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
            ax.set_yticks(range(n))
            ax.set_yticklabels(labels, fontsize=6)
            ax.set_title(short, fontsize=9)
            for i in range(n):
                for j in range(n):
                    v = mat[i, j]
                    if np.isfinite(v):
                        color = "white" if abs(v) > vmax * 0.6 else "black"
                        ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                                fontsize=4.5, color=color)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        fig.colorbar(im, cax=cbar_ax, label="ε_ij(S_k)")
        fig.suptitle(
            r"Interaction residuals $\epsilon_{ij}(S_k)$ across OCEAN judges"
            "\n"
            r"$\epsilon_{ij} = S(W+\Delta_i+\Delta_j) - S(W+\Delta_i) - S(W+\Delta_j) + S(W)$",
            fontsize=9,
        )
        fig.subplots_adjust(left=0.07, right=0.90, top=0.82, wspace=0.35)
        paper_fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(paper_fig_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] wrote paper figure: {paper_fig_path}")


# ---------------------------------------------------------------------------
# Distribution figure (main-text)
# ---------------------------------------------------------------------------

# OCEAN metric → display colour
_METRIC_COLORS = {
    "openness": "#4477AA",
    "conscientiousness": "#EE6677",
    "extraversion": "#228833",
    "agreeableness": "#CCBB44",
    "neuroticism": "#AA3377",
}


def plot_residual_distribution(
    residuals: dict[tuple[str, str], dict[str, float]],
    metrics: list[str],
    out_dir: Path,
    paper_fig_path: Path | None = None,
    outlier_threshold: float = 2.0,
) -> None:
    """1×5 panel of per-scorer histograms + KDE; top-3 outliers per panel labeled."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    short = {m: m.replace("_v2", "") for m in metrics}
    metric_order = [short[m] for m in metrics]

    # Collect per-scorer data.
    data_by_metric: dict[str, list[tuple[float, str]]] = {ms: [] for ms in metric_order}
    for (fi, fj), metric_vals in residuals.items():
        pair_label = f"{_display_label(fi)}×{_display_label(fj)}"
        for metric, eps in metric_vals.items():
            if np.isfinite(eps):
                data_by_metric[short[metric]].append((eps, pair_label))

    all_eps = [v for ms in metric_order for v, _ in data_by_metric[ms]]
    x_kde = np.linspace(min(all_eps) - 0.5, max(all_eps) + 0.5, 300)

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.13)

    from matplotlib.transforms import blended_transform_factory
    xform = blended_transform_factory(ax.transData, ax.transAxes)

    # Reference shading and zero line (drawn first, behind everything).
    ax.axvspan(-outlier_threshold, outlier_threshold,
               alpha=0.08, color="green", zorder=0, lw=0)
    ax.axvline(0, color="black", lw=0.7, ls="--", alpha=0.45, zorder=1)

    # One KDE curve + fill per scorer.
    for ms in metric_order:
        color = _METRIC_COLORS[ms]
        vals = np.array([v for v, _ in data_by_metric[ms]])
        kde = gaussian_kde(vals, bw_method="scott")
        y_kde = kde(x_kde)
        ax.plot(x_kde, y_kde, color=color, lw=2.0, label=ms, zorder=3)
        ax.fill_between(x_kde, y_kde, alpha=0.08, color=color, zorder=2)

        # Top-2 outliers per scorer: dotted vline + label at top.
        pairs = [p for _, p in data_by_metric[ms]]
        ep_arr = np.array([v for v, _ in data_by_metric[ms]])
        outlier_idx = sorted(
            [i for i, v in enumerate(ep_arr) if abs(v) > outlier_threshold],
            key=lambda i: abs(ep_arr[i]), reverse=True,
        )[:2]
        for rank, i in enumerate(outlier_idx):
            v, label = ep_arr[i], pairs[i]
            ax.axvline(v, color=color, lw=0.7, ls=":", alpha=0.55, zorder=2)
            ax.text(v, 0.97 - rank * 0.14, label,
                    fontsize=5.5, color=color,
                    ha="center", va="top", rotation=90,
                    transform=xform)

    ax.legend(fontsize=8, framealpha=0.85, loc="upper left")
    ax.set_xlabel(r"Interaction residual $\epsilon_{ij}(S_k)$", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.set_ylim(bottom=0)

    fig.suptitle(
        r"LoRA interaction residuals $\epsilon_{ij}(S_k)$ by OCEAN scorer"
        r"  —  shaded band: $|\epsilon|<2$;  top 2 outliers per scorer labeled",
        fontsize=8.5,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    if paper_fig_path is not None:
        paper_fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(paper_fig_path, bbox_inches="tight")
        print(f"[plot] wrote paper figure: {paper_fig_path}")
    local_path = out_dir / "residual_distribution.pdf"
    fig.savefig(local_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {local_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_flags() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute interaction residuals from the residuals experiment scores.",
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=SCORES_PATH,
        help=f"Path to scores.json (default: {SCORES_PATH})",
    )
    parser.add_argument(
        "--no-paper-fig",
        action="store_true",
        help="Skip writing the paper figure to paper/figures/.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    flags = _parse_flags()
    scores_path = flags.scores

    if not scores_path.exists():
        print(f"[error] scores not found: {scores_path}")
        print("Run 'python -m scripts_dev.evals.residuals_experiment.run' first.")
        sys.exit(1)

    with scores_path.open() as f:
        data = json.load(f)

    cell_scores: dict[str, dict[str, Any]] = data["cell_scores"]
    adapter_slugs: list[str] = data["adapter_slugs"]
    friendly_slugs: dict[str, str] = data["friendly_slugs"]
    metrics: list[str] = data["ocean_metrics"]
    adapter_scale: float = data.get("adapter_scale", 1.0)
    cell_metadata: dict[str, Any] = data["cell_metadata"]

    # Friendly keys in the original adapter order.
    friendly_keys = [friendly_slugs.get(slug, slug) for slug in adapter_slugs]

    # Compute residuals.
    residuals = compute_residuals(
        cell_scores, adapter_slugs, friendly_slugs, metrics, adapter_scale, cell_metadata,
    )

    n_pairs = len(residuals)
    n_finite = sum(
        1 for v in (vv for vals in residuals.values() for vv in vals.values())
        if np.isfinite(v)
    )
    print(f"[residuals] computed {n_pairs} pairs × {len(metrics)} metrics "
          f"({n_finite} finite values)")

    # Save full table.
    residuals_out = {
        f"{fi}×{fj}": dict(vals)
        for (fi, fj), vals in sorted(residuals.items())
    }
    out_path = OUTPUT_DIR / "residuals.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(residuals_out, f, indent=2)
    print(f"[residuals] wrote {out_path}")

    # Print and save summary table.
    summary = format_summary_table(residuals, metrics)
    print("\n" + summary)
    summary_path = OUTPUT_DIR / "residuals_summary.txt"
    summary_path.write_text(summary)
    print(f"\n[residuals] wrote {summary_path}")

    # Plots.
    plots_dir = OUTPUT_DIR / "plots"

    paper_fig_path: Path | None = None
    if not flags.no_paper_fig:
        try:
            from src_dev.visualisations import PAPER_FIGURES_DIR
            paper_fig_path = PAPER_FIGURES_DIR / "appendix" / "fig_residuals_heatmap.pdf"
        except ImportError:
            print("[warn] Could not import PAPER_FIGURES_DIR; skipping paper figure.")

    plot_residual_heatmaps(
        residuals,
        friendly_keys=friendly_keys,
        metrics=metrics,
        out_dir=plots_dir,
        paper_fig_path=paper_fig_path,
    )

    # Main-text figure: distribution of all 225 ε values with labeled outliers.
    dist_paper_path: Path | None = None
    if not flags.no_paper_fig:
        try:
            from src_dev.visualisations import PAPER_FIGURES_DIR
            dist_paper_path = PAPER_FIGURES_DIR / "main" / "fig_residuals_distribution.pdf"
        except ImportError:
            pass
    plot_residual_distribution(
        residuals,
        metrics=metrics,
        out_dir=plots_dir,
        paper_fig_path=dist_paper_path,
    )

    print(f"\n[done] all outputs under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
