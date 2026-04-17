"""Bootstrap confidence intervals for factor loadings.

Persona-resample bootstrap with Procrustes alignment to an anchor solution,
yielding per-cell loading CIs and a sign-stability diagnostic:

    bootstrap_loadings(data, n_factors, anchor_loadings, n_boot=500, ...)

    plot_bootstrap_loadings(result, column_defs, save_path)

Use cases:
    * Flag items whose loading sign flips across bootstrap resamples (noise).
    * Report per-factor "reliably loading" items (CI excludes 0).
    * Supply interval estimates for paper figures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src_dev.factor_analysis.congruence import procrustes_align, tucker_phi, hungarian_match
from src_dev.factor_analysis.factor_analysis import run_factor_analysis


def _tqdm(iterable, **kwargs):
    try:
        from tqdm.auto import tqdm
        return tqdm(iterable, **kwargs)
    except ImportError:
        return iterable


def _align_to_anchor(
    anchor: np.ndarray,
    other: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Procrustes-rotate ``other`` onto ``anchor``, handling sign flips.

    When ``other`` has fewer factors than anchor, pads with zeros. When more,
    selects the best-matched k anchor columns via Hungarian.

    Returns (aligned_other, mean_abs_phi_after_alignment).
    """
    k_a = anchor.shape[1]
    k_o = other.shape[1]

    if k_o == k_a:
        aligned, _ = procrustes_align(anchor, other)
    else:
        phi = tucker_phi(anchor, other)
        row_idx, col_idx, signs = hungarian_match(phi)
        # Align as many columns as we can; fill the rest with NaN so those
        # cells don't leak into CIs.
        aligned = np.full_like(anchor, np.nan)
        permuted = other[:, col_idx] * signs[None, :]
        aligned[:, row_idx] = permuted

    # Post-hoc sign flip per column to ensure +|φ|.
    for f in range(k_a):
        col = aligned[:, f]
        if not np.any(np.isfinite(col)):
            continue
        if np.dot(anchor[:, f], np.nan_to_num(col, nan=0.0)) < 0:
            aligned[:, f] = -aligned[:, f]

    # Simple alignment quality diagnostic.
    num = np.sum(anchor * np.nan_to_num(aligned, nan=0.0), axis=0)
    denom = np.linalg.norm(anchor, axis=0) * np.linalg.norm(
        np.nan_to_num(aligned, nan=0.0), axis=0,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        phi_per_factor = np.where(denom > 0, num / denom, np.nan)
    mean_abs_phi = float(np.nanmean(np.abs(phi_per_factor)))
    return aligned, mean_abs_phi


def bootstrap_loadings(
    data: np.ndarray,
    *,
    n_factors: int,
    anchor_loadings: np.ndarray,
    n_boot: int = 500,
    fa_method: str = "principal",
    rotation: str | None = "oblimin",
    seed: int = 42,
    confidence: float = 95.0,
    verbose: bool = True,
) -> dict:
    """Persona-resample bootstrap with Procrustes alignment to the anchor.

    For each of ``n_boot`` bootstrap resamples of personas:
        1. Refit FA with ``n_factors`` factors on the resampled data.
        2. Procrustes-rotate the resampled loadings onto ``anchor_loadings``.
        3. Post-hoc sign-flip columns so alignment is non-negative.
        4. Store the aligned loading matrix.

    Report per-cell percentile CIs, per-cell sign-stability (fraction of
    bootstraps whose aligned loading matches the anchor's sign), and per-item
    reliability summary (fraction of factors for which the item's anchor
    loading CI excludes 0 at ``confidence``).

    Args:
        data: [n_samples, n_vars] response matrix.
        n_factors: Must match ``anchor_loadings.shape[1]``.
        anchor_loadings: [n_vars, n_factors] reference loadings.
        n_boot: Number of bootstrap resamples.
        fa_method, rotation: FA settings. Rotate to match the anchor's
            rotation so Procrustes alignment is meaningful.
        seed: RNG seed.
        confidence: Percentile CI level (e.g. 95.0 → 2.5/97.5 bounds).
        verbose: Show tqdm progress bar when tqdm is installed.

    Returns:
        Dict with:
            anchor_loadings: [p, k] (echo of input).
            mean_loadings: [p, k] element-wise mean over successful bootstraps.
            ci_lower, ci_upper: [p, k] percentile bounds.
            sign_stability: [p, k] fraction of bootstraps agreeing in sign
                with the anchor (NaN cells excluded from denominator).
            ci_excludes_zero: [p, k] bool — CI does not bracket zero.
            n_successful_boot: number of resamples that produced a usable
                aligned matrix.
            mean_alignment_phi: mean |φ| between resampled and anchor
                loadings across successful resamples (alignment quality).
            n_boot, confidence: inputs echoed.
    """
    if anchor_loadings.shape[1] != n_factors:
        raise ValueError(
            f"anchor_loadings has {anchor_loadings.shape[1]} factors, "
            f"n_factors={n_factors}"
        )

    n_samples, n_vars = data.shape
    rng = np.random.default_rng(seed)

    stack = np.full((n_boot, n_vars, n_factors), np.nan)
    phi_quality: list[float] = []

    it = _tqdm(range(n_boot), desc="Bootstrap loadings", disable=not verbose)
    for b in it:
        idx = rng.integers(0, n_samples, size=n_samples)
        sample = data[idx]
        try:
            fa = run_factor_analysis(
                sample, n_factors=n_factors,
                method=fa_method, rotation=rotation,
            )
        except Exception:
            continue
        aligned, phi_b = _align_to_anchor(anchor_loadings, fa["loadings"])
        stack[b] = aligned
        phi_quality.append(phi_b)

    successful = ~np.all(np.isnan(stack.reshape(n_boot, -1)), axis=1)
    n_success = int(successful.sum())
    if n_success == 0:
        return {
            "anchor_loadings": anchor_loadings.tolist(),
            "mean_loadings": anchor_loadings.tolist(),
            "ci_lower": anchor_loadings.tolist(),
            "ci_upper": anchor_loadings.tolist(),
            "sign_stability": np.full_like(anchor_loadings, np.nan).tolist(),
            "ci_excludes_zero": np.zeros_like(anchor_loadings, dtype=bool).tolist(),
            "n_successful_boot": 0,
            "mean_alignment_phi": 0.0,
            "n_boot": n_boot,
            "confidence": confidence,
            "note": "All bootstrap refits failed.",
        }

    good = stack[successful]
    mean_loadings = np.nanmean(good, axis=0)
    alpha = (100.0 - confidence) / 2.0
    ci_lower = np.nanpercentile(good, alpha, axis=0)
    ci_upper = np.nanpercentile(good, 100.0 - alpha, axis=0)

    # Sign stability: fraction of successful bootstraps agreeing in sign.
    anchor_sign = np.sign(anchor_loadings)
    signs_boot = np.sign(good)
    with np.errstate(invalid="ignore"):
        agree = (signs_boot == anchor_sign[None, :, :]).astype(float)
        # Don't count NaN cells.
        agree[np.isnan(good)] = np.nan
    sign_stability = np.nanmean(agree, axis=0)

    ci_excludes_zero = (ci_lower > 0) | (ci_upper < 0)

    return {
        "anchor_loadings": anchor_loadings.tolist(),
        "mean_loadings": mean_loadings.tolist(),
        "ci_lower": ci_lower.tolist(),
        "ci_upper": ci_upper.tolist(),
        "sign_stability": sign_stability.tolist(),
        "ci_excludes_zero": ci_excludes_zero.tolist(),
        "n_successful_boot": n_success,
        "mean_alignment_phi": float(np.mean(phi_quality)) if phi_quality else 0.0,
        "n_boot": int(n_boot),
        "confidence": float(confidence),
    }


def save_bootstrap_loadings(result: dict, out_dir: Path) -> Path:
    """Write bootstrap_loadings.json alongside a summary. Returns JSON path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "bootstrap_loadings.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return path


def plot_bootstrap_loadings(
    result: dict,
    column_defs: list[dict],
    save_dir: Path,
    *,
    plt: Any | None = None,
    top_items_per_factor: int = 15,
) -> list[Path]:
    """Per-factor forest plot of loadings with bootstrap CI.

    Also writes an overview heatmap of sign-stability.

    Args:
        result: Output of ``bootstrap_loadings``.
        column_defs: Item metadata (must contain ``col_id``; optionally ``text``).
        save_dir: Directory for outputs.
        plt: Optional matplotlib module.
        top_items_per_factor: Items per forest plot, ranked by |anchor loading|.

    Returns:
        List of saved figure paths.
    """
    plt_mod = _resolve_plt(plt)
    if plt_mod is None:
        return []

    save_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    anchor = np.array(result["anchor_loadings"])
    lo = np.array(result["ci_lower"])
    hi = np.array(result["ci_upper"])
    sign_stab = np.array(result["sign_stability"])
    excludes_zero = np.array(result["ci_excludes_zero"])
    n_items, n_factors = anchor.shape
    conf = result.get("confidence", 95.0)

    item_labels = [
        (str(c.get("col_id", f"item{i}")))[:40]
        for i, c in enumerate(column_defs)
    ]

    # Per-factor forest plots.
    for f in range(n_factors):
        order = np.argsort(-np.abs(anchor[:, f]))
        top = order[: top_items_per_factor]
        loads = anchor[top, f]
        lo_f = lo[top, f]
        hi_f = hi[top, f]
        stab = sign_stab[top, f]
        excl = excludes_zero[top, f]
        labels = [item_labels[i] for i in top]

        fig, ax = plt_mod.subplots(figsize=(7, max(4, 0.28 * len(top) + 1.5)))
        ys = np.arange(len(top))
        colors = [
            "#16a34a" if (e and s >= 0.95) else
            "#2563eb" if e else
            "#f59e0b" if s >= 0.80 else
            "#dc2626"
            for e, s in zip(excl, stab)
        ]
        ax.errorbar(
            loads, ys,
            xerr=[loads - lo_f, hi_f - loads],
            fmt="none", ecolor="#4b5563", elinewidth=1, capsize=3, zorder=3,
        )
        ax.scatter(loads, ys, s=55, c=colors, edgecolors="white",
                   linewidth=0.8, zorder=4)
        ax.axvline(0, color="black", linewidth=0.6)
        for yi, s in zip(ys, stab):
            ax.text(
                ax.get_xlim()[1] if ax.get_xlim()[1] != 0 else 1.0,
                yi, f"  sign={s:.2f}", fontsize=7, va="center", color="#4b5563",
            )
        ax.set_yticks(ys)
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Factor loading")
        ax.set_title(
            f"Factor F{f} — top {len(top)} items by |loading| "
            f"(bootstrap {conf:.0f}% CI, n_boot={result['n_successful_boot']})",
            fontsize=11, fontweight="bold",
        )
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        path = save_dir / f"bootstrap_loadings_F{f}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt_mod.close(fig)
        saved.append(path)

    # Sign-stability heatmap (full matrix).
    fig, ax = plt_mod.subplots(
        figsize=(max(5, n_factors * 0.9 + 2), max(5, 0.15 * n_items + 2)),
    )
    im = ax.imshow(sign_stab, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=1.0)
    ax.set_xticks(np.arange(n_factors))
    ax.set_xticklabels([f"F{i}" for i in range(n_factors)])
    ax.set_yticks(np.arange(n_items))
    ax.set_yticklabels(item_labels, fontsize=5)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Sign-stability")
    ax.set_title(
        f"Sign stability across {result['n_successful_boot']} bootstraps",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    path = save_dir / "bootstrap_sign_stability.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt_mod.close(fig)
    saved.append(path)

    # Summary barplot: fraction of items whose CI excludes 0, per factor.
    frac_sig = np.mean(excludes_zero, axis=0)
    fig, ax = plt_mod.subplots(figsize=(max(5, n_factors * 0.7 + 1), 4))
    ax.bar(np.arange(n_factors), frac_sig, color="#2563eb",
           edgecolor="white", zorder=3)
    ax.set_xticks(np.arange(n_factors))
    ax.set_xticklabels([f"F{i}" for i in range(n_factors)])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(f"Fraction of items with CI excluding 0 (conf={conf:.0f}%)")
    ax.set_title("Per-factor loading reliability", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = save_dir / "bootstrap_fraction_reliable.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt_mod.close(fig)
    saved.append(path)

    return saved


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
