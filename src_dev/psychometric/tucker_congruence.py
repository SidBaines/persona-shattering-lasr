"""Tucker's congruence coefficient for comparing factor structures.

Tucker's φ is the standard "are these two factor solutions the same?"
metric in factor analysis. For two factor loadings vectors a, b of length
n_items,

    φ(a, b) = <a, b> / sqrt(<a, a> * <b, b>)

Standard interpretation (Lorenzo-Seva & ten Berge 2006):

    |φ| < 0.70          no similarity
    0.70 ≤ |φ| < 0.85   poor similarity
    0.85 ≤ |φ| < 0.95   fair similarity
    0.95 ≤ |φ| < 1.00   good similarity ("factorially equivalent")
    |φ| = 1.00          identical factors

Absolute value is used because factor sign is arbitrary.

This module exposes two helpers:

* :func:`tucker_phi_matrix` — full k_a × k_b matrix of |φ| for two
  loadings matrices with aligned rows (items).
* :func:`align_factors` — greedy one-to-one assignment of target factors
  to anchor factors, maximising the summed |φ|. Uses Hungarian (linear
  sum assignment) so the matching is globally optimal given the matrix.

Both helpers expect loadings in the standard ``(n_items, n_factors)``
orientation (rows = items, columns = factors).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def tucker_phi_matrix(loadings_a: np.ndarray, loadings_b: np.ndarray) -> np.ndarray:
    """Return |φ| matrix between every pair of factors in two solutions.

    Args:
        loadings_a: (n_items, k_a) loadings matrix.
        loadings_b: (n_items, k_b) loadings matrix with the same item
            order as ``loadings_a``.

    Returns:
        ``(k_a, k_b)`` array of absolute Tucker's φ. Entries lie in
        [0.0, 1.0]; higher = more similar factors.
    """
    if loadings_a.shape[0] != loadings_b.shape[0]:
        raise ValueError(
            f"Row (item) counts differ: {loadings_a.shape[0]} vs "
            f"{loadings_b.shape[0]}. Tucker's φ requires aligned items."
        )
    # column-wise norms
    norm_a = np.linalg.norm(loadings_a, axis=0)
    norm_b = np.linalg.norm(loadings_b, axis=0)
    # Guard zero-variance factors (unlikely after FA but protect).
    norm_a = np.where(norm_a == 0, np.nan, norm_a)
    norm_b = np.where(norm_b == 0, np.nan, norm_b)
    phi = (loadings_a.T @ loadings_b) / np.outer(norm_a, norm_b)
    return np.abs(phi)


@dataclass(frozen=True)
class FactorAlignment:
    """One anchor factor's best match in the target solution."""
    anchor_factor: int        # 0-based
    target_factor: int        # 0-based; -1 if no match assigned
    phi: float                # |φ| for this pair


def align_factors(
    loadings_anchor: np.ndarray,
    loadings_target: np.ndarray,
) -> list[FactorAlignment]:
    """Globally-optimal one-to-one factor matching.

    Uses the Hungarian algorithm (``scipy.optimize.linear_sum_assignment``)
    on ``-|φ|`` so we maximise summed congruence. When the two solutions
    have different factor counts, the smaller dimension bounds the number
    of matched pairs — leftover factors get ``target_factor = -1``.

    Args:
        loadings_anchor: (n_items, k_a) loadings.
        loadings_target: (n_items, k_b) loadings with matching item order.

    Returns:
        List of ``FactorAlignment`` records, one per anchor factor, in
        anchor-factor order.
    """
    from scipy.optimize import linear_sum_assignment

    phi = tucker_phi_matrix(loadings_anchor, loadings_target)
    phi_filled = np.nan_to_num(phi, nan=0.0)
    row_ind, col_ind = linear_sum_assignment(-phi_filled)
    mapping: dict[int, int] = dict(zip(row_ind.tolist(), col_ind.tolist()))

    k_a = loadings_anchor.shape[1]
    out: list[FactorAlignment] = []
    for f_a in range(k_a):
        f_b = mapping.get(f_a, -1)
        phi_val = float(phi[f_a, f_b]) if f_b >= 0 else float("nan")
        out.append(FactorAlignment(anchor_factor=f_a, target_factor=f_b, phi=phi_val))
    return out


def classify_phi(phi: float) -> str:
    """Categorical interpretation per Lorenzo-Seva & ten Berge (2006)."""
    if np.isnan(phi):
        return "n/a"
    absphi = abs(phi)
    if absphi >= 1.00:
        return "identical"
    if absphi >= 0.95:
        return "good"
    if absphi >= 0.85:
        return "fair"
    if absphi >= 0.70:
        return "poor"
    return "none"


def summarise_alignment(
    alignments_by_target: dict[str, list[FactorAlignment]],
    anchor_label: str,
) -> dict[str, Any]:
    """Flatten a dict of target→alignment into a JSON-friendly summary."""
    summary: dict[str, Any] = {
        "anchor": anchor_label,
        "matches": {},
        "per_factor_mean_phi": None,
    }
    k_factors = None
    for tgt, aligns in alignments_by_target.items():
        summary["matches"][tgt] = [
            {
                "anchor_factor": a.anchor_factor + 1,    # 1-indexed for humans
                "target_factor": a.target_factor + 1 if a.target_factor >= 0 else None,
                "phi": float(a.phi),
                "interpretation": classify_phi(a.phi),
            }
            for a in aligns
        ]
        if k_factors is None:
            k_factors = len(aligns)

    if k_factors:
        per_factor = []
        for f in range(k_factors):
            phis = [
                alignments_by_target[t][f].phi
                for t in alignments_by_target
                if not np.isnan(alignments_by_target[t][f].phi)
            ]
            per_factor.append({
                "anchor_factor": f + 1,
                "mean_phi": float(np.mean(phis)) if phis else None,
                "min_phi":  float(np.min(phis))  if phis else None,
                "max_phi":  float(np.max(phis))  if phis else None,
                "n_targets": len(phis),
            })
        summary["per_factor_mean_phi"] = per_factor

    return summary
