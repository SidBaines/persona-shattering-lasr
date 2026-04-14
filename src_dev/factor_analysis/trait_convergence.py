"""Convergent validity between factor scores and TRAIT OCEAN scores.

Spearman ρ matrix with percentile-bootstrap confidence intervals, plus a
publication-style heatmap. Caller is responsible for row-aligning
``factor_scores`` with ``trait_scores`` (same persona per row).

Pass rule: at least three of the five OCEAN traits show |ρ| ≥ 0.30 with some
factor (default ``trait_hit_threshold`` / ``min_trait_hits``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def _paired_bootstrap_corr(
    x: np.ndarray,
    y: np.ndarray,
    *,
    method: str,
    n_resamples: int,
    confidence: float,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Paired resampling bootstrap for Spearman/Pearson ρ.

    Returns (rho, ci_lower, ci_upper) using the percentile method. Returns
    NaNs for ci bounds if n < 3.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 3:
        return float("nan"), float("nan"), float("nan")

    if method == "spearman":
        rho_obs = stats.spearmanr(x, y).correlation
    elif method == "pearson":
        rho_obs = float(np.corrcoef(x, y)[0, 1])
    else:
        raise ValueError(f"method must be 'spearman' or 'pearson', got {method!r}")

    if not np.isfinite(rho_obs):
        return float("nan"), float("nan"), float("nan")

    idx_boot = rng.integers(0, n, size=(n_resamples, n))
    xs = x[idx_boot]
    ys = y[idx_boot]
    rhos = np.empty(n_resamples)
    for b in range(n_resamples):
        if method == "spearman":
            r = stats.spearmanr(xs[b], ys[b]).correlation
        else:
            r = np.corrcoef(xs[b], ys[b])[0, 1]
        rhos[b] = r if np.isfinite(r) else 0.0

    alpha = (100.0 - confidence) / 2.0
    lo = float(np.percentile(rhos, alpha))
    hi = float(np.percentile(rhos, 100.0 - alpha))
    return float(rho_obs), lo, hi


def convergent_validity(
    factor_scores: np.ndarray,
    trait_scores: np.ndarray | pd.DataFrame,
    out_dir: Path | None = None,
    *,
    trait_names: list[str] | None = None,
    factor_names: list[str] | None = None,
    method: str = "spearman",
    n_bootstrap: int = 1000,
    confidence: float = 95.0,
    trait_hit_threshold: float = 0.30,
    min_trait_hits: int = 3,
    seed: int = 42,
    plt: Any | None = None,
) -> dict:
    """Spearman ρ between factor scores and TRAIT OCEAN scores.

    Rows must be aligned: ``factor_scores[i]`` and ``trait_scores[i]`` must
    refer to the same persona. NaN cells are dropped per (factor, trait) pair.

    Args:
        factor_scores: [n, k] factor scores.
        trait_scores: [n, t] trait scores. DataFrame contributes its column
            names; numpy arrays require ``trait_names``.
        out_dir: If given, writes ``convergent_validity.json`` and a heatmap.
        trait_names: Override column names.
        factor_names: Labels for factors (defaults to "F0", "F1", ...).
        method: "spearman" (default) or "pearson".
        n_bootstrap: Bootstrap resamples per pair.
        confidence: Confidence level (e.g. 95.0).
        trait_hit_threshold: |ρ| threshold for "hit".
        min_trait_hits: Traits with a qualifying factor for ``pass``.
        seed: RNG seed.
        plt: Optional matplotlib.pyplot module.

    Returns:
        Dict with correlation matrix, CI matrices, best-matches, pass flag.
    """
    if isinstance(trait_scores, pd.DataFrame):
        if trait_names is None:
            trait_names = list(trait_scores.columns)
        trait_values = trait_scores.values.astype(float)
    else:
        trait_values = np.asarray(trait_scores, dtype=float)
        if trait_names is None:
            trait_names = [f"trait_{j}" for j in range(trait_values.shape[1])]

    if factor_scores.shape[0] != trait_values.shape[0]:
        raise ValueError(
            f"Row count mismatch: factor_scores has {factor_scores.shape[0]}, "
            f"trait_scores has {trait_values.shape[0]}. Caller must align."
        )

    n, k = factor_scores.shape
    t = trait_values.shape[1]
    if factor_names is None:
        factor_names = [f"F{i}" for i in range(k)]

    rho_mat = np.full((t, k), np.nan)
    lo_mat = np.full((t, k), np.nan)
    hi_mat = np.full((t, k), np.nan)
    n_valid_mat = np.zeros((t, k), dtype=int)

    rng = np.random.default_rng(seed)
    for j in range(t):
        y = trait_values[:, j]
        for i in range(k):
            x = factor_scores[:, i]
            # Draw a fresh sub-rng so (trait, factor) pair ordering doesn't
            # change individual CIs.
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
            rho, lo, hi = _paired_bootstrap_corr(
                x, y, method=method,
                n_resamples=n_bootstrap,
                confidence=confidence,
                rng=pair_rng,
            )
            rho_mat[j, i] = rho
            lo_mat[j, i] = lo
            hi_mat[j, i] = hi
            n_valid_mat[j, i] = int(np.sum(np.isfinite(x) & np.isfinite(y)))

    # Best-factor-per-trait and best-trait-per-factor.
    abs_rho = np.abs(rho_mat)
    best_factor_per_trait = []
    for j in range(t):
        if np.all(np.isnan(abs_rho[j])):
            best_factor_per_trait.append(None)
            continue
        i = int(np.nanargmax(abs_rho[j]))
        best_factor_per_trait.append({
            "factor_index": i, "factor_name": factor_names[i],
            "rho": float(rho_mat[j, i]),
            "ci_lower": float(lo_mat[j, i]), "ci_upper": float(hi_mat[j, i]),
        })

    best_trait_per_factor = []
    for i in range(k):
        if np.all(np.isnan(abs_rho[:, i])):
            best_trait_per_factor.append(None)
            continue
        j = int(np.nanargmax(abs_rho[:, i]))
        best_trait_per_factor.append({
            "trait_index": j, "trait_name": trait_names[j],
            "rho": float(rho_mat[j, i]),
            "ci_lower": float(lo_mat[j, i]), "ci_upper": float(hi_mat[j, i]),
        })

    # Pass criterion: count traits with at least one |ρ| ≥ threshold.
    trait_hits = []
    for j in range(t):
        row = abs_rho[j]
        hit = bool(np.any(row >= trait_hit_threshold)) if np.any(np.isfinite(row)) else False
        trait_hits.append(hit)
    n_hits = int(sum(trait_hits))
    passed = n_hits >= min_trait_hits

    result = {
        "method": method,
        "n_rows": int(n),
        "trait_names": list(trait_names),
        "factor_names": list(factor_names),
        "rho_matrix": rho_mat.tolist(),
        "ci_lower_matrix": lo_mat.tolist(),
        "ci_upper_matrix": hi_mat.tolist(),
        "n_valid_matrix": n_valid_mat.tolist(),
        "best_factor_per_trait": best_factor_per_trait,
        "best_trait_per_factor": best_trait_per_factor,
        "trait_hit_threshold": trait_hit_threshold,
        "min_trait_hits": min_trait_hits,
        "trait_hits": trait_hits,
        "n_trait_hits": n_hits,
        "n_bootstrap": n_bootstrap,
        "confidence": confidence,
        "pass": passed,
    }

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "convergent_validity.json", "w") as f:
            json.dump(result, f, indent=2)
        plot_convergent_heatmap(
            result, out_dir / "convergent_validity.png", plt=plt,
        )

    traits_with_hit = [
        n for n, hit in zip(trait_names, trait_hits) if hit
    ]
    print(
        f"  Convergent validity ({method}): "
        f"{n_hits}/{t} traits hit (|ρ|≥{trait_hit_threshold:.2f}): "
        f"{traits_with_hit} ({'PASS' if passed else 'FAIL'})"
    )
    return result


def plot_convergent_heatmap(
    result: dict,
    out_path: Path,
    *,
    plt: Any | None = None,
) -> Path | None:
    """Heatmap of ρ with CI annotations.

    Cell value is ρ; cell annotation is "ρ [lo, hi]". Traits on the y-axis,
    factors on the x-axis. Diverging colormap centered at zero.
    """
    plt_mod = _resolve_plt(plt)
    if plt_mod is None:
        return None

    rho = np.array(result["rho_matrix"])
    lo = np.array(result["ci_lower_matrix"])
    hi = np.array(result["ci_upper_matrix"])
    trait_names = result["trait_names"]
    factor_names = result["factor_names"]
    thr = result["trait_hit_threshold"]

    t, k = rho.shape
    fig, ax = plt_mod.subplots(
        figsize=(max(6, 1.1 * k + 3), max(4, 0.7 * t + 1.5))
    )
    vmax = max(0.3, float(np.nanmax(np.abs(rho))) if np.any(np.isfinite(rho)) else 0.3)
    im = ax.imshow(rho, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    for jj in range(t):
        for ii in range(k):
            if not np.isfinite(rho[jj, ii]):
                continue
            bold = abs(rho[jj, ii]) >= thr
            txt = f"{rho[jj, ii]:+.2f}\n[{lo[jj, ii]:+.2f}, {hi[jj, ii]:+.2f}]"
            ax.text(
                ii, jj, txt, ha="center", va="center",
                fontsize=7 if bold else 6.5,
                fontweight="bold" if bold else "normal",
                color="white" if abs(rho[jj, ii]) > 0.55 * vmax else "#111111",
            )

    ax.set_xticks(np.arange(k))
    ax.set_xticklabels(factor_names, fontsize=10, rotation=0)
    ax.set_yticks(np.arange(t))
    ax.set_yticklabels(trait_names, fontsize=10)
    ax.set_title(
        f"Convergent validity ({result['method']}, "
        f"bootstrap {int(result['confidence'])}% CI, n={result['n_rows']})",
        fontsize=11, fontweight="bold",
    )
    fig.colorbar(im, ax=ax, shrink=0.8, label=f"{result['method']} ρ")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt_mod.close(fig)
    return out_path


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
