"""Factor analysis wrapper around the factor_analyzer library."""

from __future__ import annotations

import numpy as np

# Patch sklearn compatibility: factor_analyzer 0.5.1 uses the deprecated
# 'force_all_finite' kwarg which was renamed to 'ensure_all_finite' in sklearn 1.8.
import sklearn.utils.validation as _skl_validation

_orig_check_array = _skl_validation.check_array


def _patched_check_array(*args, **kwargs):
    if "force_all_finite" in kwargs:
        kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
    return _orig_check_array(*args, **kwargs)


_skl_validation.check_array = _patched_check_array

from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo
import factor_analyzer.factor_analyzer as _fa_mod

_fa_mod.check_array = _patched_check_array


def adequacy_tests(data: np.ndarray) -> dict:
    """Run Bartlett's test of sphericity and Kaiser-Meyer-Olkin (KMO) test.

    These test whether factor analysis is appropriate for the data.
    - Bartlett's: tests if the correlation matrix differs from identity.
      Significant p-value (< 0.05) means FA is appropriate.
    - KMO: measures sampling adequacy. Values > 0.6 are acceptable, > 0.8 good.

    Note: On PCA-reduced data these tests may give uninformative results
    (PCA scores are uncorrelated by construction). Run on the original
    standardized data for meaningful results.

    Args:
        data: Input matrix [n_samples, n_vars].

    Returns:
        Dict with bartlett_chi2, bartlett_p, kmo_per_variable, kmo_overall.
    """
    chi2, p = calculate_bartlett_sphericity(data)
    kmo_per_var, kmo_overall = calculate_kmo(data)

    print(f"Bartlett's test: chi2={chi2:.1f}, p={p:.2e}")
    print(f"KMO: {kmo_overall:.3f}")

    return {
        "bartlett_chi2": float(chi2),
        "bartlett_p": float(p),
        "kmo_per_variable": kmo_per_var,
        "kmo_overall": float(kmo_overall),
    }


def run_factor_analysis(
    data: np.ndarray,
    n_factors: int,
    method: str = "principal",
    rotation: str | None = "varimax",
) -> dict:
    """Run factor analysis using the factor_analyzer library.

    Args:
        data: Input matrix [n_samples, n_vars].
        n_factors: Number of factors to extract.
        method: Extraction method — "principal", "minres", or "ml".
        rotation: Rotation method — "varimax", "promax", "oblimin", or None.

    Returns:
        Dict with keys:
            loadings: Factor loading matrix [n_vars, n_factors].
            scores: Factor scores [n_samples, n_factors].
            communalities: Communality per variable [n_vars].
            eigenvalues_original: Eigenvalues of the original correlation matrix.
            eigenvalues_common: Eigenvalues of the common factor solution.
            ss_loadings: Sum of squared loadings per factor.
            proportion_variance: Proportion of variance per factor.
            rotation_matrix: Rotation matrix if rotation was applied, else None.
    """
    fa = FactorAnalyzer(
        n_factors=n_factors,
        method=method,
        rotation=rotation,
    )
    fa.fit(data)

    loadings = fa.loadings_
    scores = fa.transform(data)
    communalities = fa.get_communalities()
    ev_original, ev_common = fa.get_eigenvalues()
    variance = fa.get_factor_variance()
    # get_factor_variance returns (ss_loadings, proportion_variance, cumulative_variance)

    rotation_matrix = getattr(fa, "rotation_matrix_", None)

    print(f"Factor analysis: {n_factors} factors, method={method}, rotation={rotation}")
    print(f"  Loadings shape: {loadings.shape}")
    print(f"  Mean communality: {communalities.mean():.4f}")
    print(f"  Total variance explained: {variance[2][-1]:.3%}")

    return {
        "loadings": loadings,
        "scores": scores,
        "communalities": communalities,
        "eigenvalues_original": ev_original,
        "eigenvalues_common": ev_common,
        "ss_loadings": variance[0],
        "proportion_variance": variance[1],
        "cumulative_variance": variance[2],
        "rotation_matrix": rotation_matrix,
    }
