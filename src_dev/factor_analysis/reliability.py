"""Reliability statistics for factor analysis results.

Provides ICC (intraclass correlation coefficient) computation for assessing
test-retest reliability of factor scores across grouped observations.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def compute_icc(
    scores: np.ndarray,
    metadata: list[dict],
    n_factors: int,
    *,
    group_field: str = "input_group_id",
    fallback_field: str = "sample_id",
    confidence: float = 0.95,
) -> dict:
    """Compute ICC(1) and ICC(k) per factor from grouped observations.

    Uses a one-way random-effects ANOVA decomposition. Groups are defined by
    *group_field* in the metadata (typically the seed prompt). Only groups with
    >= 2 members contribute.

    Args:
        scores: Factor scores [n_samples, n_factors].
        metadata: Per-sample metadata dicts (must contain *group_field*).
        n_factors: Number of factors (columns of *scores* to use).
        group_field: Metadata key that defines the grouping.
        fallback_field: Used when *group_field* is absent from a metadata entry.
        confidence: Confidence level for CIs (default 0.95).

    Returns:
        Dict with keys:
            icc1: List of ICC(1) values per factor.
            icc1_ci_lower: List of lower CI bounds per factor.
            icc1_ci_upper: List of upper CI bounds per factor.
            icc_k: List of ICC(k) (Spearman-Brown corrected) per factor.
            f_values: List of F-statistics per factor.
            n_groups: Number of groups with >= 2 members.
            n_total: Total observations across those groups.
            mean_group_size: Arithmetic mean group size.
    """
    # Group indices
    group_to_indices: dict[str, list[int]] = {}
    for i, meta in enumerate(metadata):
        gid = meta.get(group_field, meta.get(fallback_field, str(i)))
        group_to_indices.setdefault(gid, []).append(i)

    multi_groups = {
        gid: idxs
        for gid, idxs in group_to_indices.items()
        if len(idxs) >= 2
    }

    if len(multi_groups) < 5:
        return {
            "icc1": [],
            "icc1_ci_lower": [],
            "icc1_ci_upper": [],
            "icc_k": [],
            "f_values": [],
            "n_groups": len(multi_groups),
            "n_total": sum(len(idxs) for idxs in multi_groups.values()),
            "mean_group_size": 0.0,
            "error": f"Too few groups with >= 2 members ({len(multi_groups)})",
        }

    alpha = 1.0 - confidence

    icc1_values: list[float] = []
    icc1_ci_lower: list[float] = []
    icc1_ci_upper: list[float] = []
    icc_k_values: list[float] = []
    f_values: list[float] = []

    for fi in range(n_factors):
        group_scores: list[list[float]] = []
        for idxs in multi_groups.values():
            g = [scores[i, fi] for i in idxs if i < scores.shape[0]]
            if len(g) >= 2:  # only include groups that still have >=2 after bounds check
                group_scores.append(g)
        if len(group_scores) < 5:
            # Not enough valid groups for this factor
            icc1_values.append(0.0)
            icc1_ci_lower.append(0.0)
            icc1_ci_upper.append(0.0)
            icc_k_values.append(0.0)
            f_values.append(0.0)
            continue

        n_groups = len(group_scores)
        group_sizes = [len(g) for g in group_scores]
        n_total = sum(group_sizes)
        # Harmonic mean group size (more accurate for unbalanced designs)
        k_harmonic = n_groups / sum(1.0 / ni for ni in group_sizes)

        group_means = [np.mean(g) for g in group_scores]
        grand_mean = np.mean([s for g in group_scores for s in g])

        # Between-group MS
        ss_between = sum(
            ni * (gm - grand_mean) ** 2
            for ni, gm in zip(group_sizes, group_means)
        )
        df_between = n_groups - 1
        ms_between = ss_between / df_between

        # Within-group MS
        ss_within = sum(
            sum((x - gm) ** 2 for x in g)
            for g, gm in zip(group_scores, group_means)
        )
        df_within = n_total - n_groups
        ms_within = ss_within / df_within if df_within > 0 else 1e-12

        # F-statistic
        f_val = ms_between / ms_within if ms_within > 0 else 0.0
        f_values.append(float(f_val))

        # ICC(1)
        denom = ms_between + (k_harmonic - 1) * ms_within
        icc1 = (ms_between - ms_within) / denom if denom > 0 else 0.0
        icc1_values.append(float(icc1))

        # ICC(k) — Spearman-Brown corrected
        icc_k = (k_harmonic * icc1) / (1 + (k_harmonic - 1) * icc1) if (1 + (k_harmonic - 1) * icc1) > 0 else 0.0
        icc_k_values.append(float(icc_k))

        # F-based confidence interval for ICC(1)
        # Shrout & Fleiss (1979), McGraw & Wong (1996)
        f_lower = stats.f.ppf(alpha / 2, df_between, df_within)
        f_upper = stats.f.ppf(1 - alpha / 2, df_between, df_within)

        # CI bounds: transform the F-distribution CI on F into ICC space
        icc_lo = (f_val / f_upper - 1) / (f_val / f_upper + k_harmonic - 1)
        icc_hi = (f_val / f_lower - 1) / (f_val / f_lower + k_harmonic - 1)
        icc1_ci_lower.append(float(icc_lo))
        icc1_ci_upper.append(float(icc_hi))

    n_total_all = sum(len(idxs) for idxs in multi_groups.values())
    return {
        "icc1": icc1_values,
        "icc1_ci_lower": icc1_ci_lower,
        "icc1_ci_upper": icc1_ci_upper,
        "icc_k": icc_k_values,
        "f_values": f_values,
        "n_groups": len(multi_groups),
        "n_total": n_total_all,
        "mean_group_size": n_total_all / len(multi_groups) if multi_groups else 0.0,
    }
