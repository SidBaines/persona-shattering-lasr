"""Diagnostic plots for factor analysis results.

``plot_fa_visualisations`` is the high-level entry point — it produces the
full set of plots for one FA fit (one (method, n_factors, rotation) triple)
under ``save_dir``. Individual plots are exposed so subset scripts can call
them independently.

Plots:
    1_loading_heatmap.png       — items × factors heatmap, clustered by dominant factor
    2_score_scatter_matrix.png  — pairwise factor-score scatter plots
    3_communalities.png         — per-item communality bars
    4_score_distributions.png   — per-factor score histograms + KDE + normality tests
    5_factor_correlations.png   — inter-factor correlation heatmap (oblique rotations only)
    7_scores_by_archetype.png   — factor scores grouped by interviewer archetype
    8_prompt_icc.png            — within-prompt vs between-prompt variance (ICC(1))
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def plot_fa_visualisations(
    fa_result: dict,
    column_defs: list[dict],
    metadata: list[dict],
    data: np.ndarray,
    label: str,
    save_dir: Path,
    *,
    rollout_dir: Path | None = None,
) -> None:
    """Generate a suite of diagnostic visualisations for factor analysis results.

    Args:
        fa_result: Dict produced by ``src_dev.factor_analysis.run_factor_analysis``.
        column_defs: Per-column definitions (same length as loadings.shape[0]).
        metadata: Per-row metadata (same length as scores.shape[0]).
        data: Unused — retained for signature compatibility.
        label: Short description used in plot titles and filenames.
        save_dir: Directory to write plots into (created if needed).
        rollout_dir: Directory containing ``archetype_assignments.json``.
            Required for ``7_scores_by_archetype`` — the plot is skipped when
            ``rollout_dir`` is None or the file is missing.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    loadings = fa_result["loadings"]  # [n_items, n_factors]
    scores = fa_result["scores"]      # [n_personas, n_factors]
    communalities = fa_result["communalities"]  # [n_items]
    n_factors = loadings.shape[1]

    print(f"\n  [Viz] Generating plots for {label} ({n_factors} factors)...")

    plot_loading_heatmap(loadings, column_defs, n_factors, save_dir, label, plt, TwoSlopeNorm)
    plot_score_scatter_matrix(scores, n_factors, save_dir, label, plt)
    plot_communalities(communalities, column_defs, save_dir, label, plt)
    plot_score_distributions(scores, n_factors, save_dir, label, plt)
    plot_factor_correlations(fa_result, n_factors, save_dir, label, plt, TwoSlopeNorm)
    plot_scores_by_archetype(scores, metadata, n_factors, save_dir, label, plt, rollout_dir=rollout_dir)
    plot_prompt_icc(scores, metadata, n_factors, save_dir, label, plt)

    print(f"  [Viz] All plots saved to {save_dir}")


def plot_loading_heatmap(
    loadings: np.ndarray,
    column_defs: list[dict],
    n_factors: int,
    save_dir: Path,
    label: str,
    plt,
    TwoSlopeNorm,
) -> None:
    """Loading heatmap: items (rows) × factors (columns), clustered by dominant factor."""
    n_items = loadings.shape[0]

    dominant_factor = np.argmax(np.abs(loadings), axis=1)
    sort_keys = []
    for i in range(n_items):
        df = dominant_factor[i]
        sort_keys.append((df, -np.abs(loadings[i, df])))
    sort_order = sorted(range(n_items), key=lambda i: sort_keys[i])

    sorted_loadings = loadings[sort_order]
    sorted_labels = []
    for idx in sort_order:
        cd = column_defs[idx]
        text = cd["text"][:55]
        block_tag = cd["block"][:2].upper()
        rev = "(R) " if cd.get("reverse_keyed", False) else ""
        sorted_labels.append(f"[{block_tag}] {rev}{text}")

    fig_h = max(6, n_items * 0.22)
    fig_w = max(6, 3 + n_factors * 0.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    vmax = max(0.8, np.max(np.abs(sorted_loadings)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(sorted_loadings, aspect="auto", cmap="RdBu_r", norm=norm)

    ax.set_xticks(range(n_factors))
    ax.set_xticklabels([f"F{i}" for i in range(n_factors)], fontsize=10)
    ax.set_yticks(range(n_items))
    ax.set_yticklabels(sorted_labels, fontsize=max(5, min(8, 200 / n_items)))

    prev_df = dominant_factor[sort_order[0]]
    for row_i, orig_idx in enumerate(sort_order):
        df = dominant_factor[orig_idx]
        if df != prev_df:
            ax.axhline(row_i - 0.5, color="black", linewidth=0.8, alpha=0.5)
            prev_df = df

    fig.colorbar(im, ax=ax, label="Loading", shrink=0.6)
    ax.set_title(f"Factor Loadings — {label}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Factor")
    fig.tight_layout()
    fig.savefig(save_dir / "1_loading_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    1_loading_heatmap.png")


def plot_score_scatter_matrix(
    scores: np.ndarray,
    n_factors: int,
    save_dir: Path,
    label: str,
    plt,
) -> None:
    """Pairwise scatter plots of factor scores.

    For ≤15 factors, shows the full NxN matrix. For >15 factors, shows the top
    20 most correlated pairs (by absolute correlation) to keep the plot readable.
    """
    if n_factors < 2:
        return

    MAX_FULL_MATRIX = 15
    MAX_PAIRS = 20

    if n_factors <= MAX_FULL_MATRIX:
        fig, axes = plt.subplots(
            n_factors, n_factors,
            figsize=(3 * n_factors, 3 * n_factors),
        )
        if n_factors == 1:
            axes = np.array([[axes]])

        for i in range(n_factors):
            for j in range(n_factors):
                ax = axes[i, j]
                if i == j:
                    ax.hist(scores[:, i], bins=30, color="#2563eb", alpha=0.7, edgecolor="white")
                    ax.set_ylabel("Count" if j == 0 else "")
                elif i > j:
                    ax.scatter(
                        scores[:, j], scores[:, i],
                        alpha=0.15, s=8, color="#2563eb", edgecolors="none",
                    )
                    r = np.corrcoef(scores[:, j], scores[:, i])[0, 1]
                    ax.annotate(
                        f"r={r:.2f}", xy=(0.05, 0.92), xycoords="axes fraction",
                        fontsize=9, fontweight="bold",
                        color="#dc2626" if abs(r) > 0.3 else "#6b7280",
                    )
                else:
                    ax.set_visible(False)

                if i == n_factors - 1:
                    ax.set_xlabel(f"F{j}", fontsize=10)
                if j == 0 and i != j:
                    ax.set_ylabel(f"F{i}", fontsize=10)

        fig.suptitle(f"Factor Score Scatter Matrix — {label}", fontsize=14, fontweight="bold", y=1.01)
    else:
        corr_matrix = np.corrcoef(scores.T)
        pairs = []
        for i in range(n_factors):
            for j in range(i + 1, n_factors):
                pairs.append((abs(corr_matrix[i, j]), i, j))
        pairs.sort(reverse=True)
        top_pairs = pairs[:MAX_PAIRS]

        n_show = len(top_pairs)
        cols = min(4, n_show)
        rows = (n_show + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)

        for idx, (abs_r, fi, fj) in enumerate(top_pairs):
            ax = axes[idx // cols, idx % cols]
            ax.scatter(
                scores[:, fj], scores[:, fi],
                alpha=0.15, s=8, color="#2563eb", edgecolors="none",
            )
            r = corr_matrix[fi, fj]
            ax.annotate(
                f"r={r:.2f}", xy=(0.05, 0.92), xycoords="axes fraction",
                fontsize=10, fontweight="bold",
                color="#dc2626" if abs(r) > 0.3 else "#6b7280",
            )
            ax.set_xlabel(f"F{fj}", fontsize=10)
            ax.set_ylabel(f"F{fi}", fontsize=10)
            ax.set_title(f"F{fi} vs F{fj}", fontsize=10)

        for idx in range(n_show, rows * cols):
            axes[idx // cols, idx % cols].set_visible(False)

        fig.suptitle(
            f"Top {n_show} Factor Score Pairs (by |r|) — {label}",
            fontsize=14, fontweight="bold", y=1.01,
        )

    fig.tight_layout()
    fig.savefig(save_dir / "2_score_scatter_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    2_score_scatter_matrix.png")


def plot_communalities(
    communalities: np.ndarray,
    column_defs: list[dict],
    save_dir: Path,
    label: str,
    plt,
) -> None:
    """Per-item communality bar chart, sorted by communality."""
    n_items = len(communalities)
    order = np.argsort(communalities)[::-1]

    item_labels = []
    for idx in order:
        cd = column_defs[idx]
        text = cd["text"][:50]
        block_tag = cd["block"][:2].upper()
        item_labels.append(f"[{block_tag}] {text}")

    fig_h = max(5, n_items * 0.2)
    fig, ax = plt.subplots(figsize=(7, fig_h))

    colors = ["#2563eb" if c >= 0.2 else "#dc2626" for c in communalities[order]]
    y_pos = np.arange(n_items)
    ax.barh(y_pos, communalities[order], color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(item_labels, fontsize=max(5, min(8, 200 / n_items)))
    ax.invert_yaxis()
    ax.axvline(0.2, color="#dc2626", linestyle="--", linewidth=0.8, alpha=0.6, label="h²=0.2 threshold")
    ax.set_xlabel("Communality (h²)")
    ax.set_title(f"Communalities — {label}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(save_dir / "3_communalities.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    3_communalities.png")


def plot_score_distributions(
    scores: np.ndarray,
    n_factors: int,
    save_dir: Path,
    label: str,
    plt,
) -> None:
    """Per-factor score distribution histograms with KDE overlay."""
    cols = min(n_factors, 4)
    rows = (n_factors + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), squeeze=False)

    for fi in range(n_factors):
        ax = axes[fi // cols, fi % cols]
        s = scores[:, fi]
        ax.hist(s, bins=40, color="#2563eb", alpha=0.6, edgecolor="white", density=True)

        from scipy.stats import gaussian_kde
        try:
            kde = gaussian_kde(s)
            x_grid = np.linspace(s.min() - 0.5, s.max() + 0.5, 200)
            ax.plot(x_grid, kde(x_grid), color="#dc2626", linewidth=1.5)
        except Exception:
            pass

        from scipy.stats import shapiro
        try:
            sub = s[:500] if len(s) > 500 else s
            _, p_val = shapiro(sub)
            ax.annotate(
                f"Shapiro p={p_val:.3f}",
                xy=(0.95, 0.92), xycoords="axes fraction",
                fontsize=8, ha="right",
                color="#059669" if p_val > 0.05 else "#dc2626",
            )
        except Exception:
            pass

        from scipy.stats import skew, kurtosis
        sk = skew(s)
        ku = kurtosis(s)
        ax.annotate(
            f"skew={sk:.2f}  kurt={ku:.2f}",
            xy=(0.95, 0.82), xycoords="axes fraction",
            fontsize=7, ha="right", color="#6b7280",
        )

        ax.set_title(f"Factor {fi}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Score")

    for i in range(n_factors, rows * cols):
        axes[i // cols, i % cols].set_visible(False)

    fig.suptitle(f"Factor Score Distributions — {label}", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(save_dir / "4_score_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    4_score_distributions.png")


def plot_factor_correlations(
    fa_result: dict,
    n_factors: int,
    save_dir: Path,
    label: str,
    plt,
    TwoSlopeNorm,
) -> None:
    """Inter-factor correlation heatmap (oblique rotations only)."""
    phi = fa_result.get("factor_correlation_matrix")
    if phi is None:
        # Fallback: empirical factor-score correlation (≈identity for varimax).
        phi = np.corrcoef(fa_result["scores"].T)

    if n_factors < 2:
        return

    fig, ax = plt.subplots(figsize=(max(4, n_factors * 0.8 + 2), max(4, n_factors * 0.8 + 1)))
    vmax = max(0.5, np.max(np.abs(phi - np.eye(n_factors))))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(phi, cmap="RdBu_r", norm=norm)
    for i in range(n_factors):
        for j in range(n_factors):
            ax.text(j, i, f"{phi[i, j]:.2f}", ha="center", va="center",
                    fontsize=10, fontweight="bold" if i != j and abs(phi[i, j]) > 0.3 else "normal")

    ax.set_xticks(range(n_factors))
    ax.set_xticklabels([f"F{i}" for i in range(n_factors)])
    ax.set_yticks(range(n_factors))
    ax.set_yticklabels([f"F{i}" for i in range(n_factors)])
    fig.colorbar(im, ax=ax, label="Correlation", shrink=0.7)
    ax.set_title(f"Factor Correlations — {label}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_dir / "5_factor_correlations.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    5_factor_correlations.png")


def plot_scores_by_archetype(
    scores: np.ndarray,
    metadata: list[dict],
    n_factors: int,
    save_dir: Path,
    label: str,
    plt,
    *,
    rollout_dir: Path | None = None,
) -> None:
    """Violin plots of factor scores grouped by interviewer archetype.

    Requires ``rollout_dir/archetype_assignments.json``. Silently skipped
    when ``rollout_dir`` is None or the file is missing.
    """
    if rollout_dir is None:
        print(f"    7_scores_by_archetype.png — skipped (no rollout_dir supplied)")
        return
    assignments_path = Path(rollout_dir) / "archetype_assignments.json"
    if not assignments_path.exists():
        print(f"    7_scores_by_archetype.png — skipped (no archetype_assignments.json)")
        return

    with open(assignments_path) as f:
        sample_to_archetype: dict[str, str] = json.load(f)

    archetypes = []
    valid_mask = []
    for i, meta in enumerate(metadata):
        arch = sample_to_archetype.get(meta["sample_id"])
        archetypes.append(arch)
        valid_mask.append(arch is not None)

    valid_mask = np.array(valid_mask)
    if valid_mask.sum() < 10:
        print(f"    7_scores_by_archetype.png — skipped (too few matched rows)")
        return

    valid_scores = scores[valid_mask]
    valid_archetypes = [a for a, v in zip(archetypes, valid_mask) if v]

    unique_archetypes = sorted(set(valid_archetypes))
    n_archetypes = len(unique_archetypes)
    arch_to_idx = {a: i for i, a in enumerate(unique_archetypes)}

    cols = min(n_factors, 3)
    rows = (n_factors + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for fi in range(n_factors):
        ax = axes[fi // cols, fi % cols]

        grouped = [[] for _ in range(n_archetypes)]
        for score_val, arch in zip(valid_scores[:, fi], valid_archetypes):
            grouped[arch_to_idx[arch]].append(score_val)

        parts = ax.violinplot(
            grouped, positions=range(n_archetypes),
            showmeans=True, showmedians=True,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor("#2563eb")
            pc.set_alpha(0.5)
        parts["cmeans"].set_color("#dc2626")
        parts["cmedians"].set_color("#059669")

        ax.set_xticks(range(n_archetypes))
        ax.set_xticklabels(
            [a[:15] for a in unique_archetypes],
            rotation=45, ha="right", fontsize=8,
        )
        ax.set_title(f"Factor {fi}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Score")
        ax.grid(axis="y", alpha=0.3)

    for i in range(n_factors, rows * cols):
        axes[i // cols, i % cols].set_visible(False)

    fig.suptitle(
        f"Factor Scores by Interviewer Archetype — {label}",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(save_dir / "7_scores_by_archetype.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    7_scores_by_archetype.png")


def plot_prompt_icc(
    scores: np.ndarray,
    metadata: list[dict],
    n_factors: int,
    save_dir: Path,
    label: str,
    plt,
) -> None:
    """ICC(1) per factor: proportion of factor-score variance attributable to the seed prompt.

    High ICC means the prompt drives the factor more than stochastic rollout variation.
    Low ICC means the factor captures genuine run-to-run behavioural variation.
    """
    from src_dev.factor_analysis.reliability import compute_icc

    icc_result = compute_icc(scores, metadata, n_factors)

    if icc_result.get("error"):
        print(f"    8_prompt_icc.png — skipped ({icc_result['error']})")
        return

    icc_values = icc_result["icc1"]
    n_groups = icc_result["n_groups"]

    fig, ax = plt.subplots(figsize=(max(5, n_factors * 1.2 + 1), 4))
    x = np.arange(n_factors)
    colors = ["#f59e0b" if v > 0.5 else "#2563eb" for v in icc_values]
    bars = ax.bar(x, icc_values, color=colors, edgecolor="white", width=0.6)

    for bar, val in zip(bars, icc_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"F{i}" for i in range(n_factors)], fontsize=11)
    ax.set_ylim(0, min(1.0, max(icc_values) * 1.3 + 0.05))
    ax.set_ylabel("ICC(1)")
    ax.set_xlabel("Factor")
    ax.axhline(0.5, color="#f59e0b", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.annotate("ICC=0.5", xy=(n_factors - 0.5, 0.51), fontsize=8, color="#f59e0b")
    ax.set_title(
        f"Prompt ICC — {label}\n"
        f"({n_groups} prompts with ≥2 rollouts)",
        fontsize=13, fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "8_prompt_icc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    8_prompt_icc.png")

    with open(save_dir / "8_prompt_icc.json", "w") as f:
        json.dump({
            "icc_per_factor": icc_values,
            "n_groups": n_groups,
            "n_total_personas": icc_result["n_total"],
        }, f, indent=2)
