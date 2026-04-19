"""Trait-aware visualisations of FA loadings.

A second pass of loading plots that superimposes the OCEAN trait identity of
each item (from ``primary_dimension``) on top of the factor structure. Used
both for the raw pooled FA and the trait-oriented FA pass (which conditions
on trait_mcq items so the trait identity is denser).

Five plots per fit (keyed ``6a..6e``):
    6a_trait_sorted_loading_heatmap.png
    6b_per_trait_loading_distributions.png
    6c_cumulative_top_k_composition.png
    6d_per_trait_abs_loading_ecdf.png
    6e_signed_loading_strip.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# Fixed OCEAN color palette so trait colors are consistent across plots.
OCEAN_TRAIT_COLORS: dict[str, str] = {
    "openness": "#4c78a8",
    "conscientiousness": "#59a14f",
    "extraversion": "#e45756",
    "agreeableness": "#b279a2",
    "neuroticism": "#f28e2b",
}
FALLBACK_TRAIT_COLOR = "#9ca3af"


def trait_color(trait: str) -> str:
    return OCEAN_TRAIT_COLORS.get(trait, FALLBACK_TRAIT_COLOR)


def canonical_trait_order(item_dims: list[str]) -> list[str]:
    """Canonical OCEAN order, restricted to traits that actually appear.

    Extra (non-OCEAN) dimensions are appended alphabetically so plots stay
    stable as new questionnaires add dimensions.
    """
    ocean = [
        "openness",
        "conscientiousness",
        "extraversion",
        "agreeableness",
        "neuroticism",
    ]
    present = [t for t in ocean if t in item_dims]
    extras = sorted(set(item_dims) - set(present))
    return present + extras


def plot_trait_aware_fa_visualisations(
    loadings: np.ndarray,
    item_dims: list[str],
    save_dir: Path,
    label: str,
    *,
    top_k: int = 20,
    signed_caveat: str | None = None,
) -> None:
    """Render five trait-aware views of the FA loadings matrix.

    Args:
        loadings: [n_items × n_factors] float.
        item_dims: primary_dimension for each row of ``loadings``. Items with
            ``None`` dimension are dropped from all plots.
        save_dir: Directory to write PNGs into.
        label: Label for plot titles (e.g. "raw_varimax", "trait_oriented_varimax").
        top_k: K for top-K cumulative composition curves (also annotated on them).
        signed_caveat: If set, appended to signed-loading plot titles (e.g. a
            letter-encoded-matrix warning).
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    loadings = np.asarray(loadings)
    mask = np.array([d is not None for d in item_dims])
    if not mask.any():
        print(f"  [TraitViz] No items with dimensions for {label} — skipping.")
        return
    loadings = loadings[mask]
    item_dims = [d for d, m in zip(item_dims, mask) if m]
    trait_order = canonical_trait_order(item_dims)

    plot_trait_sorted_loading_heatmap(
        loadings, item_dims, trait_order,
        save_dir / "6a_trait_sorted_loading_heatmap.png",
        label=label, signed_caveat=signed_caveat,
    )
    plot_per_trait_loading_distributions(
        loadings, item_dims, trait_order,
        save_dir / "6b_per_trait_loading_distributions.png",
        label=label, signed_caveat=signed_caveat,
    )
    plot_cumulative_top_k_composition(
        loadings, item_dims, trait_order,
        save_dir / "6c_cumulative_top_k_composition.png",
        label=label, top_k_marker=top_k,
    )
    plot_per_trait_loading_ecdfs(
        loadings, item_dims, trait_order,
        save_dir / "6d_per_trait_abs_loading_ecdf.png",
        label=label,
    )
    plot_signed_loading_strip(
        loadings, item_dims, trait_order,
        save_dir / "6e_signed_loading_strip.png",
        label=label, signed_caveat=signed_caveat,
    )


def plot_trait_sorted_loading_heatmap(
    loadings: np.ndarray,
    item_dims: list[str],
    trait_order: list[str],
    save_path: Path,
    *,
    label: str,
    signed_caveat: str | None = None,
) -> None:
    """Items × Factors heatmap sorted by OCEAN trait block, with separators."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    n_items, n_factors = loadings.shape
    trait_rank = {t: i for i, t in enumerate(trait_order)}
    dominant_factor = np.argmax(np.abs(loadings), axis=1)
    keys = [
        (
            trait_rank.get(item_dims[i], len(trait_order)),
            int(dominant_factor[i]),
            -float(np.abs(loadings[i, dominant_factor[i]])),
        )
        for i in range(n_items)
    ]
    order = sorted(range(n_items), key=lambda i: keys[i])
    sorted_loadings = loadings[order]
    sorted_dims = [item_dims[i] for i in order]

    fig_h = max(6, n_items * 0.14)
    fig_w = max(6, 3 + n_factors * 0.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    vmax = max(0.5, float(np.max(np.abs(sorted_loadings))))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(sorted_loadings, aspect="auto", cmap="RdBu_r", norm=norm)

    ax.set_xticks(range(n_factors))
    ax.set_xticklabels([f"F{i+1}" for i in range(n_factors)], fontsize=10)

    # Row labels: compact trait abbreviation + row index within trait block.
    row_labels: list[str] = []
    per_trait_counter: dict[str, int] = {}
    for d in sorted_dims:
        per_trait_counter[d] = per_trait_counter.get(d, 0) + 1
        row_labels.append(f"{d[:4]}_{per_trait_counter[d]:02d}")
    ax.set_yticks(range(n_items))
    ax.set_yticklabels(row_labels, fontsize=max(4, min(8, 220 / n_items)))

    # Trait-block separators + side labels.
    prev = sorted_dims[0]
    block_start = 0
    for i, d in enumerate(sorted_dims + [None]):
        if d != prev:
            ax.axhline(i - 0.5, color="black", linewidth=1.2, alpha=0.75)
            mid = (block_start + i - 1) / 2
            color = trait_color(prev)
            ax.text(
                -0.6, mid, prev, ha="right", va="center",
                fontsize=10, fontweight="bold", color=color,
                transform=ax.get_yaxis_transform(),
            )
            block_start = i
            prev = d

    title = f"Factor loadings, trait-sorted — {label}"
    if signed_caveat:
        title += f"\n({signed_caveat})"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Factor")
    fig.colorbar(im, ax=ax, label="Loading", shrink=0.6)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    6a_trait_sorted_loading_heatmap.png")


def plot_per_trait_loading_distributions(
    loadings: np.ndarray,
    item_dims: list[str],
    trait_order: list[str],
    save_path: Path,
    *,
    label: str,
    signed_caveat: str | None = None,
) -> None:
    """Per-factor violin+strip plots of signed item loadings grouped by trait.

    One subplot per factor. Within each subplot, one violin+strip per trait
    showing the full distribution of that trait's items' loadings (not just
    top-K).
    """
    import matplotlib.pyplot as plt

    n_items, n_factors = loadings.shape
    dims_arr = np.array(item_dims)
    n_cols = min(3, n_factors)
    n_rows = int(np.ceil(n_factors / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.2 * n_cols, 3.0 * n_rows),
        squeeze=False,
    )
    vmax = float(np.max(np.abs(loadings)))
    lim = max(0.4, vmax * 1.1)
    for f, ax in zip(range(n_factors), axes.flat):
        col = loadings[:, f]
        data_per_trait = [col[dims_arr == t] for t in trait_order]
        positions = np.arange(len(trait_order))
        vp = ax.violinplot(
            [d for d in data_per_trait if len(d) > 0],
            positions=[p for p, d in zip(positions, data_per_trait) if len(d) > 0],
            widths=0.75, showmeans=False, showmedians=True, showextrema=False,
        )
        for body, t in zip(vp["bodies"], [t for t, d in zip(trait_order, data_per_trait) if len(d) > 0]):
            body.set_facecolor(trait_color(t))
            body.set_edgecolor("black")
            body.set_alpha(0.35)
        rng = np.random.default_rng(0)
        for pos, t, d in zip(positions, trait_order, data_per_trait):
            if len(d) == 0:
                continue
            jitter = rng.uniform(-0.18, 0.18, size=len(d))
            ax.scatter(
                pos + jitter, d, color=trait_color(t),
                s=14, alpha=0.7, edgecolor="white", linewidth=0.4,
            )
        ax.axhline(0, color="#6b7280", linewidth=0.8, alpha=0.6)
        ax.set_xticks(positions)
        ax.set_xticklabels([t[:4] for t in trait_order], fontsize=9)
        ax.set_ylim(-lim, lim)
        ax.set_title(f"F{f+1}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Signed loading" if f % n_cols == 0 else "")
        ax.grid(axis="y", alpha=0.3, linewidth=0.4)

    for ax in axes.flat[n_factors:]:
        ax.axis("off")

    suptitle = f"Per-trait loading distributions — {label}"
    if signed_caveat:
        suptitle += f"  ({signed_caveat})"
    fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    6b_per_trait_loading_distributions.png")


def plot_cumulative_top_k_composition(
    loadings: np.ndarray,
    item_dims: list[str],
    trait_order: list[str],
    save_path: Path,
    *,
    label: str,
    top_k_marker: int = 20,
) -> None:
    """Cumulative top-K composition: for each factor, one line per trait
    showing how many top-K items (sorted by |loading|) belong to that trait
    as K sweeps from 1 to n_items. Dashed lines show the uniform baseline
    (expected count if loadings were unrelated to trait)."""
    import matplotlib.pyplot as plt

    n_items, n_factors = loadings.shape
    dims_arr = np.array(item_dims)
    trait_fracs = {t: float(np.mean(dims_arr == t)) for t in trait_order}

    n_cols = min(3, n_factors)
    n_rows = int(np.ceil(n_factors / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.2 * n_cols, 3.0 * n_rows),
        squeeze=False,
    )
    K_arr = np.arange(1, n_items + 1)
    for f, ax in zip(range(n_factors), axes.flat):
        col = loadings[:, f]
        rank = np.argsort(-np.abs(col))
        dim_sequence = dims_arr[rank]
        for t in trait_order:
            is_trait = (dim_sequence == t).astype(int)
            cum = np.cumsum(is_trait)
            ax.plot(
                K_arr, cum, color=trait_color(t), linewidth=1.8,
                label=t,
            )
            ax.plot(
                K_arr, K_arr * trait_fracs[t],
                color=trait_color(t), linewidth=0.8, linestyle="--", alpha=0.6,
            )
        if 1 <= top_k_marker <= n_items:
            ax.axvline(
                top_k_marker, color="#6b7280", linestyle=":", linewidth=1,
                alpha=0.8,
            )
            ax.text(
                top_k_marker, ax.get_ylim()[1] * 0.98,
                f" K={top_k_marker}", fontsize=8, color="#374151",
                va="top", ha="left",
            )
        ax.set_title(f"F{f+1}", fontsize=11, fontweight="bold")
        ax.set_xlabel("K (items ranked by |loading|)")
        if f % n_cols == 0:
            ax.set_ylabel("Cumulative count")
        ax.grid(alpha=0.3, linewidth=0.4)

    handles = [
        plt.Line2D([0], [0], color=trait_color(t), linewidth=2, label=t)
        for t in trait_order
    ]
    handles.append(plt.Line2D([0], [0], color="#6b7280", linestyle="--",
                              linewidth=1, label="expected (uniform)"))
    fig.legend(handles=handles, loc="lower center",
               ncol=min(6, len(handles)), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    for ax in axes.flat[n_factors:]:
        ax.axis("off")

    fig.suptitle(f"Cumulative top-K trait composition — {label}",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    6c_cumulative_top_k_composition.png")


def plot_per_trait_loading_ecdfs(
    loadings: np.ndarray,
    item_dims: list[str],
    trait_order: list[str],
    save_path: Path,
    *,
    label: str,
) -> None:
    """Per-factor ECDFs of |loading|, one curve per trait.

    A factor that captures trait X will have that trait's ECDF shifted to
    the right (more mass at high |loading|) vs other traits.
    """
    import matplotlib.pyplot as plt

    n_items, n_factors = loadings.shape
    dims_arr = np.array(item_dims)
    n_cols = min(3, n_factors)
    n_rows = int(np.ceil(n_factors / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.2 * n_cols, 3.0 * n_rows),
        squeeze=False,
    )
    vmax = float(np.max(np.abs(loadings)))
    for f, ax in zip(range(n_factors), axes.flat):
        col_abs = np.abs(loadings[:, f])
        for t in trait_order:
            vals = np.sort(col_abs[dims_arr == t])
            if len(vals) == 0:
                continue
            ys = np.arange(1, len(vals) + 1) / len(vals)
            ax.plot(vals, ys, color=trait_color(t), linewidth=1.8, label=t)
        ax.set_title(f"F{f+1}", fontsize=11, fontweight="bold")
        ax.set_xlabel("|loading|")
        if f % n_cols == 0:
            ax.set_ylabel("ECDF (fraction ≤)")
        ax.set_xlim(0, max(0.4, vmax * 1.05))
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3, linewidth=0.4)

    handles = [
        plt.Line2D([0], [0], color=trait_color(t), linewidth=2, label=t)
        for t in trait_order
    ]
    fig.legend(handles=handles, loc="lower center",
               ncol=min(5, len(handles)), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    for ax in axes.flat[n_factors:]:
        ax.axis("off")

    fig.suptitle(f"Per-trait ECDFs of |loading| — {label}",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    6d_per_trait_abs_loading_ecdf.png")


def plot_signed_loading_strip(
    loadings: np.ndarray,
    item_dims: list[str],
    trait_order: list[str],
    save_path: Path,
    *,
    label: str,
    signed_caveat: str | None = None,
) -> None:
    """Per-factor horizontal strip plot: every item's signed loading, grouped
    by trait on the y-axis. Unlike the violin view, this emphasizes sign and
    keeps individual items visible."""
    import matplotlib.pyplot as plt

    n_items, n_factors = loadings.shape
    dims_arr = np.array(item_dims)
    n_cols = min(3, n_factors)
    n_rows = int(np.ceil(n_factors / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.6 * n_cols, 0.35 * len(trait_order) * n_rows + 1.2),
        squeeze=False,
    )
    vmax = float(np.max(np.abs(loadings)))
    lim = max(0.4, vmax * 1.1)
    rng = np.random.default_rng(1)
    for f, ax in zip(range(n_factors), axes.flat):
        col = loadings[:, f]
        for y, t in enumerate(trait_order):
            vals = col[dims_arr == t]
            if len(vals) == 0:
                continue
            jitter = rng.uniform(-0.22, 0.22, size=len(vals))
            ax.scatter(
                vals, np.full_like(vals, y) + jitter,
                color=trait_color(t), s=22, alpha=0.75,
                edgecolor="white", linewidth=0.4,
            )
            ax.scatter(
                [float(np.mean(vals))], [y], marker="|",
                color="black", s=180, linewidth=1.6, zorder=5,
            )
        ax.axvline(0, color="#6b7280", linewidth=0.8, alpha=0.7)
        ax.set_yticks(range(len(trait_order)))
        ax.set_yticklabels(trait_order, fontsize=9)
        ax.set_ylim(-0.7, len(trait_order) - 0.3)
        ax.set_xlim(-lim, lim)
        ax.set_xlabel("Signed loading")
        ax.set_title(f"F{f+1}", fontsize=11, fontweight="bold")
        ax.grid(axis="x", alpha=0.3, linewidth=0.4)
    for ax in axes.flat[n_factors:]:
        ax.axis("off")

    suptitle = f"Signed-loading strip by trait — {label}"
    if signed_caveat:
        suptitle += f"  ({signed_caveat})"
    fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    6e_signed_loading_strip.png")
