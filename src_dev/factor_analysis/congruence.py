"""Compare two factor-analysis solutions.

Primitives:
    tucker_phi(L_a, L_b)       : Tucker's congruence coefficient matrix.
    procrustes_align(L_ref, L) : orthogonal Procrustes rotation of L onto L_ref.
    hungarian_match(sim)       : optimal row↔col matching on |sim|.

High-level:
    compare_solutions(L_a, L_b, *, align="procrustes") -> SolutionComparison
        — full pipeline: Procrustes (optional) → Hungarian → per-pair φ.

Plots:
    plot_paired_loading_heatmap : side-by-side loading heatmap, matched columns.
    plot_phi_bar                : bar chart of per-matched-pair φ.

Interpretation (Lorenzo-Seva & ten Berge 2006):
    φ ≥ 0.95  essentially identical factors
    φ ≥ 0.85  "fair" similarity
    φ < 0.85  factors should not be considered equal
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

_log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# PRIMITIVES
# ═════════════════════════════════════════════════════════════════════════════


def tucker_phi(L_a: np.ndarray, L_b: np.ndarray) -> np.ndarray:
    """Tucker's congruence coefficient between every pair of columns.

    φ(a, b) = Σᵢ aᵢ bᵢ / √(Σᵢ aᵢ² · Σᵢ bᵢ²) — i.e. cosine similarity on raw
    loadings (not centered). Canonical factor-comparison metric.

    Args:
        L_a: Loadings [n_items, k_a].
        L_b: Loadings [n_items, k_b]. Must share the first dimension with L_a.

    Returns:
        [k_a, k_b] matrix of signed congruence coefficients in [-1, 1].
    """
    if L_a.shape[0] != L_b.shape[0]:
        raise ValueError(
            f"Loading matrices must share item axis: {L_a.shape[0]} vs {L_b.shape[0]}"
        )
    a_norm = np.linalg.norm(L_a, axis=0, keepdims=True)  # [1, k_a]
    b_norm = np.linalg.norm(L_b, axis=0, keepdims=True)  # [1, k_b]
    a_safe = np.where(a_norm > 0, a_norm, 1.0)
    b_safe = np.where(b_norm > 0, b_norm, 1.0)
    L_a_n = L_a / a_safe
    L_b_n = L_b / b_safe
    return L_a_n.T @ L_b_n  # [k_a, k_b]


def procrustes_align(
    L_ref: np.ndarray,
    L_other: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Orthogonal Procrustes rotation of L_other onto L_ref.

    Finds the orthogonal R that minimizes ‖L_other @ R − L_ref‖_F. Requires
    L_other and L_ref to share both axes (use ``hungarian_match`` alone for
    k-mismatched comparisons).

    Args:
        L_ref: Reference loadings [n_items, k].
        L_other: Loadings to rotate [n_items, k].

    Returns:
        Tuple of (L_other_aligned [n_items, k], rotation matrix R [k, k]).
    """
    if L_ref.shape != L_other.shape:
        raise ValueError(
            f"procrustes_align requires identical shapes, got "
            f"{L_ref.shape} and {L_other.shape}. Use hungarian_match for "
            f"mismatched k."
        )
    # M = L_other.T @ L_ref  (k × k);  R = U V.T.
    M = L_other.T @ L_ref
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    return L_other @ R, R


def hungarian_match(sim: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimal row→column matching on |sim| via the Hungarian algorithm.

    Maximizes Σ |sim[r, c]| over the matched pairs. Handles rectangular sim
    (matches min(k_a, k_b) pairs).

    Args:
        sim: Similarity matrix [k_a, k_b]. Signed values allowed; matching
            uses absolute value, and the sign of the matched entry is
            returned as the sign flip.

    Returns:
        (row_idx, col_idx, sign_flips), each of length min(k_a, k_b).
        ``sim[row_idx[i], col_idx[i]]`` is the matched (signed) congruence
        for the i-th pair; ``sign_flips[i]`` is ±1.
    """
    cost = -np.abs(sim)
    row_idx, col_idx = linear_sum_assignment(cost)
    matched = sim[row_idx, col_idx]
    sign_flips = np.where(matched >= 0, 1.0, -1.0)
    return row_idx, col_idx, sign_flips


# ═════════════════════════════════════════════════════════════════════════════
# HIGH-LEVEL COMPARISON
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class SolutionComparison:
    """Result of comparing two factor solutions.

    Attributes:
        phi_matched: [n_matched] signed Tucker φ on each matched pair (after
            any Procrustes rotation).
        matched_a_indices: [n_matched] column indices into L_a.
        matched_b_indices: [n_matched] column indices into L_b.
        sign_flips: [n_matched] ±1 per matched pair.
        loadings_a: [n_items, k_a] — original L_a (passed through for plotting).
        loadings_b_aligned: [n_items, n_matched] — L_b columns reordered to
            match L_a's column order on matched pairs, sign-flipped so
            congruence is non-negative. Columns of L_a without a match (when
            k_a > k_b) are not represented; vice versa unmatched L_b columns
            are dropped.
        full_phi_matrix: [k_a, k_b] — full Tucker φ matrix post-rotation,
            pre-matching (useful for diagnostics / split-merge analysis).
        procrustes_rotation: [k, k] rotation or None if align="hungarian".
        align_method: "procrustes" or "hungarian".
        n_matched: min(k_a, k_b).
    """

    phi_matched: np.ndarray
    matched_a_indices: np.ndarray
    matched_b_indices: np.ndarray
    sign_flips: np.ndarray
    loadings_a: np.ndarray
    loadings_b_aligned: np.ndarray
    full_phi_matrix: np.ndarray
    procrustes_rotation: np.ndarray | None
    align_method: str
    n_matched: int

    @property
    def mean_phi(self) -> float:
        return float(np.mean(np.abs(self.phi_matched)))

    @property
    def median_phi(self) -> float:
        return float(np.median(np.abs(self.phi_matched)))

    @property
    def min_phi(self) -> float:
        return float(np.min(np.abs(self.phi_matched)))


def compare_solutions(
    L_a: np.ndarray,
    L_b: np.ndarray,
    *,
    align: str = "procrustes",
) -> SolutionComparison:
    """Compare two factor solutions: rotate (optional), match, compute φ.

    When k_a == k_b and align="procrustes": orthogonal Procrustes rotates L_b
    onto L_a, then Hungarian matches columns on |φ|. This is the preferred
    comparison for same-k solutions (e.g. split stability, phrasing sweeps).

    When k_a != k_b (or align="hungarian"): Hungarian matching runs directly
    on the Tucker φ matrix. Use for k±1 sensitivity analysis.

    Args:
        L_a: Reference loadings [n_items, k_a].
        L_b: Loadings to compare [n_items, k_b].
        align: "procrustes" (requires k_a == k_b) or "hungarian".

    Returns:
        SolutionComparison with matched pairs, aligned loadings, and the full
        φ matrix.
    """
    if align not in ("procrustes", "hungarian"):
        raise ValueError(f"align must be 'procrustes' or 'hungarian', got {align!r}")

    k_a = L_a.shape[1]
    k_b = L_b.shape[1]

    if align == "procrustes" and k_a != k_b:
        # Fall back to hungarian rather than erroring — many callers discover
        # k-mismatch only after fitting. Warn so the caller knows the result
        # isn't rotation-aligned, only index-matched.
        _log.warning(
            "compare_solutions: k mismatch (k_a=%d, k_b=%d) — falling back to "
            "'hungarian' alignment. The returned comparison is Hungarian-only, "
            "not Procrustes-rotated.",
            k_a, k_b,
        )
        align = "hungarian"

    if align == "procrustes":
        L_b_rot, R = procrustes_align(L_a, L_b)
    else:
        L_b_rot = L_b
        R = None

    phi_full = tucker_phi(L_a, L_b_rot)  # [k_a, k_b]
    row_idx, col_idx, sign_flips = hungarian_match(phi_full)

    # Build aligned loadings: columns ordered so matched pair i corresponds to
    # column i of L_a's row_idx[i]. If the row_idx is not already in ascending
    # 0..n-1 order (e.g. k_a > k_b), we preserve the match order as-is.
    order = np.argsort(row_idx)
    row_idx_sorted = row_idx[order]
    col_idx_sorted = col_idx[order]
    sign_flips_sorted = sign_flips[order]

    L_b_aligned = L_b_rot[:, col_idx_sorted] * sign_flips_sorted[None, :]
    phi_matched = phi_full[row_idx_sorted, col_idx_sorted] * sign_flips_sorted

    return SolutionComparison(
        phi_matched=phi_matched,
        matched_a_indices=row_idx_sorted,
        matched_b_indices=col_idx_sorted,
        sign_flips=sign_flips_sorted,
        loadings_a=L_a,
        loadings_b_aligned=L_b_aligned,
        full_phi_matrix=phi_full,
        procrustes_rotation=R,
        align_method=align,
        n_matched=len(row_idx),
    )


# ═════════════════════════════════════════════════════════════════════════════
# PLOTS
# ═════════════════════════════════════════════════════════════════════════════


def _resolve_plt(plt: Any | None) -> Any | None:
    if plt is not None:
        return plt
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt_mod
        return plt_mod
    except ImportError:
        return None


def plot_paired_loading_heatmap(
    cmp: SolutionComparison,
    item_labels: list[str],
    out_path: Path,
    *,
    label_a: str = "Solution A",
    label_b: str = "Solution B",
    plt: Any | None = None,
) -> Path | None:
    """Side-by-side loading heatmaps for matched factor pairs.

    Each panel is an [n_items, n_matched] heatmap. Panels are aligned so
    column *i* in both panels is the *i*-th matched pair (L_b is
    Procrustes-rotated + sign-flipped to align with L_a).

    Args:
        cmp: SolutionComparison from compare_solutions.
        item_labels: [n_items] labels for the item axis.
        out_path: File to write (PNG).
        label_a, label_b: Titles for the two panels.
        plt: Optional matplotlib.pyplot module.

    Returns:
        Path to written file, or None if matplotlib unavailable.
    """
    plt_mod = _resolve_plt(plt)
    if plt_mod is None:
        return None

    La = cmp.loadings_a[:, cmp.matched_a_indices]
    Lb = cmp.loadings_b_aligned
    n_items, n_matched = La.shape
    vmax = float(max(np.abs(La).max(), np.abs(Lb).max()))

    fig_h = max(4, 0.12 * n_items + 1.5)
    fig_w = max(6, 1.4 * n_matched + 3)
    fig, axes = plt_mod.subplots(
        1, 2, figsize=(fig_w, fig_h), sharey=True, constrained_layout=True,
    )

    for ax, mat, title in zip(axes, (La, Lb), (label_a, label_b)):
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(np.arange(n_matched))
        ax.set_xticklabels(
            [f"F{i}\n(φ={cmp.phi_matched[i]:+.2f})" for i in range(n_matched)],
            fontsize=8,
        )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Matched factor")

    axes[0].set_yticks(np.arange(n_items))
    axes[0].set_yticklabels(item_labels, fontsize=6)
    axes[0].set_ylabel("Item")

    fig.colorbar(im, ax=axes, shrink=0.6, label="Loading")
    fig.suptitle(
        f"Paired loading heatmap — {cmp.align_method}, "
        f"mean |φ|={cmp.mean_phi:.3f}",
        fontsize=12, fontweight="bold",
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt_mod.close(fig)
    return out_path


def plot_phi_bar(
    cmp: SolutionComparison,
    out_path: Path,
    *,
    title: str | None = None,
    plt: Any | None = None,
) -> Path | None:
    """Bar chart of per-matched-pair Tucker φ with Lorenzo-Seva reference lines.

    Args:
        cmp: SolutionComparison from compare_solutions.
        out_path: File to write (PNG).
        title: Optional plot title override.
        plt: Optional matplotlib.pyplot module.

    Returns:
        Path to written file, or None if matplotlib unavailable.
    """
    plt_mod = _resolve_plt(plt)
    if plt_mod is None:
        return None

    phi = np.abs(cmp.phi_matched)
    n = len(phi)
    colors = [
        "#16a34a" if v >= 0.95 else "#2563eb" if v >= 0.85 else "#dc2626"
        for v in phi
    ]

    fig, ax = plt_mod.subplots(figsize=(max(5, 0.9 * n + 1.5), 4.5))
    ax.bar(np.arange(n), phi, color=colors, edgecolor="white", width=0.6, zorder=3)
    ax.axhline(0.95, color="#16a34a", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axhline(0.85, color="#2563eb", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(n - 0.4, 0.955, "identical (0.95)", fontsize=7, color="#16a34a", ha="right")
    ax.text(n - 0.4, 0.855, "fair (0.85)", fontsize=7, color="#2563eb", ha="right")
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(
        [f"F{cmp.matched_a_indices[i]}↔{cmp.matched_b_indices[i]}" for i in range(n)],
        fontsize=9,
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("|Tucker φ|")
    ax.set_xlabel("Matched factor pair")
    ax.set_title(
        title or f"Per-factor congruence ({cmp.align_method}), mean |φ|={cmp.mean_phi:.3f}",
        fontsize=11, fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt_mod.close(fig)
    return out_path
