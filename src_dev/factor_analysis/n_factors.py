"""Methods for determining the number of factors to retain.

Horn's permutation parallel analysis lives in ``parallel_analysis.py`` and is
the pipeline's default selector. This module provides complementary methods,
reported alongside Horn's for triangulation:

    velicer_map(data)
        Velicer's Minimum Average Partial. Partials out k principal components
        from the correlation matrix and tracks the mean squared off-diagonal
        partial correlation across k; chooses the k at the minimum. Tends to
        be more conservative than Horn's (under-retain vs. over-retain).

    empirical_kaiser_criterion(data)
        Braeken & van Assen (2017). Closed-form upper bound on eigenvalues
        under the null of no additional factors, derived from the
        Marchenko–Pastur distribution. Adjusts Kaiser's eigenvalue-> 1 rule
        for finite-sample bias.

    acceleration_factor(eigenvalues)
        Raîche, Roipel & Blais (2006) objective scree. Picks the k at which
        the second difference of the eigenvalue sequence is maximized — the
        "knee" of the scree plot without subjective elbow calling.

    kaiser_guttman(eigenvalues)
        Classic "eigenvalue > 1 on the correlation matrix" rule. Known to
        over-extract; included only for comparison/transparency.

    optimal_coordinates(eigenvalues)
        Raîche et al. (2013) non-graphical Cattell scree. For each k, linearly
        forecast λ_k from (λ_{k+1}, λ_p); retain the leading run of k's whose
        real λ exceeds its forecast. Global scree-shape criterion.

    scree_elbow(eigenvalues)
        Kneedle-style programmatic elbow: k at the point furthest (perpendicular
        distance) below the chord between the first and last eigenvalue.
        Differs from acceleration_factor (local 2nd difference) in picking up
        global curvature.

    cv_reconstruction_k(data, k_max, n_folds, ...)
        Held-out persona Gaussian negative log-likelihood under the k-factor
        model, averaged across K folds. Picks k = argmin mean NLL. Literature
        standard out-of-sample fit criterion for FA (Browne 1979; Bro et al.
        2008 for component models). Connects directly to the
        replicability/predictivity question: does a k-factor model generalize
        to unseen personas better than k-1 or k+1?

    suggest_n_factors(data, methods=..., ...)
        Orchestrator. Runs every requested method and returns a dict keyed by
        method name with each method's k recommendation + diagnostics.

    plot_n_factors_comparison(result, save_path)
        Visual comparison of every method's k on a single figure (real-data
        eigenvalues, each method's threshold/curve, and its chosen k).

Future direction — ML-based methods (not implemented here)
    With ``method="ml"`` extraction (instead of PAF / principal), the factor
    model is a proper multivariate-normal likelihood model, which unlocks:

        * BIC / AIC across k: compute the ML log-likelihood at each k and pick
          the k that minimizes BIC (penalized log-likelihood). Standard model
          selection, but sensitive to violations of the MVN assumption — our
          columns are ordinal/bounded, so BIC will favor too-complex models
          unless we polychoric-correlate first.
        * Chi-square test of fit: H0 is "k factors is sufficient". LRT against
          the saturated model. Overpowers at large n.
        * RMSEA / TLI / CFI: SEM-style fit indices. RMSEA < 0.06 and CFI >
          0.95 are conventional thresholds. Most defensible when reported as
          an ensemble.

    Not currently wired in because the pipeline uses ``method="principal"``
    (PAF) throughout, and switching to ML for k-selection-only introduces
    a two-extraction inconsistency + ML convergence failures on wide data.
    When revisited, add an ``ml_bic`` / ``ml_fit_indices`` method here and
    gate it on a ``supports_ml: bool`` check for the configured extraction.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from src_dev.factor_analysis.factor_analysis import run_factor_analysis

_log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# CORRELATION / EIGENVALUE HELPERS
# ═════════════════════════════════════════════════════════════════════════════


def _corr_safe(data: np.ndarray) -> np.ndarray:
    """Correlation matrix that tolerates zero-variance columns (→ identity row).

    Zero-variance columns are given a tiny independent noise ridge so
    ``np.corrcoef`` doesn't produce NaNs. The noise is seeded so that
    repeated calls on the same input are reproducible; only the affected
    columns are perturbed.
    """
    std = data.std(axis=0, ddof=0)
    zero_cols = np.flatnonzero(std == 0)
    if zero_cols.size > 0:
        data = data.copy()
        rng = np.random.default_rng(0)
        n = data.shape[0]
        data[:, zero_cols] = data[:, zero_cols] + 1e-12 * rng.standard_normal((n, zero_cols.size))
    return np.corrcoef(data, rowvar=False)


def _sorted_eigvals_desc(mat: np.ndarray) -> np.ndarray:
    """Descending eigenvalues of a symmetric matrix."""
    return np.sort(np.linalg.eigvalsh(mat))[::-1]


# ═════════════════════════════════════════════════════════════════════════════
# VELICER'S MAP
# ═════════════════════════════════════════════════════════════════════════════


_MAP_DEFAULT_K_MAX = 15


def velicer_map(data: np.ndarray, k_max: int | None = None) -> dict:
    """Velicer's Minimum Average Partial test.

    Algorithm (Velicer 1976, revised 2000):
        1. Z-score columns; compute correlation matrix R.
        2. Eigendecompose R.
        3. For k = 0, 1, ..., k_max:
            a. Partial out the first k principal components from R.
            b. Re-standardize the partialed covariance to a correlation
               (divide by diag sqrt).
            c. fm(k) = mean of squared off-diagonal entries.
            d. fm4(k) = same but 4th power (Velicer's revised statistic,
               which down-weights noise).
        4. k̂ = argmin fm (standard MAP) and argmin fm4 (revised MAP).

    Args:
        data: [n_samples, n_vars] response matrix.
        k_max: Maximum k to test. Defaults to min(n_vars − 2, 15) — matches
            the orchestrator default so standalone calls and orchestrated
            calls agree.

    Returns:
        Dict with keys ``fm``, ``fm4`` (both length k_max+1; fm[0] is the
        mean squared off-diagonal of R itself), ``n_recommended`` (argmin fm,
        revised MAP recommendation), and ``n_recommended_fm4`` (argmin fm4).
        ``semantics="argmin"`` — n_recommended is the k at the argmin of the
        statistic, and may equal k_max if the curve is still descending
        (caller should check saturation).
    """
    n_samples, n_vars = data.shape
    if k_max is None:
        k_max = min(n_vars - 2, _MAP_DEFAULT_K_MAX)
    if k_max < 1:
        return {
            "fm": [], "fm4": [], "n_recommended": 0, "n_recommended_fm4": 0,
            "semantics": "argmin", "saturated": False,
        }

    R = _corr_safe(data)
    # Eigendecompose R; columns of V are eigenvectors, sorted descending by λ.
    eigvals, eigvecs = np.linalg.eigh(R)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    fm_list: list[float] = []
    fm4_list: list[float] = []

    # k = 0: no partialing — average squared off-diagonal of R itself.
    off_diag_mask = ~np.eye(n_vars, dtype=bool)

    def _mean_sq_offdiag(mat: np.ndarray, power: int) -> float:
        return float(np.mean(mat[off_diag_mask] ** power))

    fm_list.append(_mean_sq_offdiag(R, 2))
    fm4_list.append(_mean_sq_offdiag(R, 4))

    # For k >= 1, form the partial covariance:
    #   U_k  = eigvecs[:, :k] · diag(sqrt(λ_{1..k}))  (component loading matrix)
    #   cov_resid_k = R − U_k U_k.T
    #   R_k = cov_resid_k / sqrt(diag ⊗ diag)
    for k in range(1, k_max + 1):
        if eigvals[k - 1] <= 0:
            # Can't partial out a non-positive eigenvalue; stop early.
            break
        U = eigvecs[:, :k] * np.sqrt(np.maximum(eigvals[:k], 0.0))[None, :]
        cov_resid = R - U @ U.T
        d = np.diag(cov_resid)
        if np.any(d <= 0):
            # Numerical: residual variance collapsed. Stop — extracting more
            # components is meaningless.
            break
        scale = np.sqrt(d)
        R_k = cov_resid / np.outer(scale, scale)
        np.fill_diagonal(R_k, 0.0)  # zero diagonal so it doesn't contribute.
        fm_list.append(_mean_sq_offdiag(R_k, 2))
        fm4_list.append(_mean_sq_offdiag(R_k, 4))

    fm_arr = np.array(fm_list, dtype=float)
    fm4_arr = np.array(fm4_list, dtype=float)

    # n_recommended is the k at minimum. fm[0] corresponds to k=0 (retain 0).
    # Typically MAP is reported as argmin on k >= 1 — retaining zero factors
    # isn't usable. Return argmin over k >= 1 when available, falling back to
    # 0 when the min really is at k=0 (data is already near-diagonal).
    def _argmin_recommendation(arr: np.ndarray) -> int:
        return int(np.argmin(arr))  # 0-indexed over k values 0..len-1

    n_rec = _argmin_recommendation(fm_arr)
    n_rec_fm4 = _argmin_recommendation(fm4_arr)
    k_max_tested = len(fm_arr) - 1
    saturated = bool(
        (n_rec == k_max_tested and k_max_tested >= 1) or
        (n_rec_fm4 == k_max_tested and k_max_tested >= 1)
    )

    return {
        "fm": fm_arr.tolist(),
        "fm4": fm4_arr.tolist(),
        "n_recommended": n_rec,
        "n_recommended_fm4": n_rec_fm4,
        "k_max_tested": k_max_tested,
        "semantics": "argmin",
        "saturated": saturated,
    }


# ═════════════════════════════════════════════════════════════════════════════
# EMPIRICAL KAISER CRITERION
# ═════════════════════════════════════════════════════════════════════════════


def empirical_kaiser_criterion(data: np.ndarray) -> dict:
    """Empirical Kaiser Criterion (Braeken & van Assen 2017).

    For each k, the reference eigenvalue λ_ref(k) under the null of no
    additional factors is derived from the Marchenko–Pastur distribution
    (asymptotic eigenvalue density of Wishart correlation matrices). EKC
    retains the **leading run** of k's for which:

        λ_real(k) > max( λ_ref(k),  1 )

    with λ_ref(k) = (1 + √(p / n))² · (p − Σ_{j<k} λ_real(j)) / (p − k + 1).
    This adapts for variance already accounted for at lower k, fixing the
    Kaiser rule's tendency to over-retain.

    Note: the leading-run convention (stop at the first failing index) is the
    one Braeken & van Assen recommend; counting *total* eigenvalues above
    threshold can sawtooth when a late rubble eigenvalue randomly bumps
    above its ref.

    Args:
        data: [n_samples, n_vars] response matrix.

    Returns:
        Dict with ``real_eigenvalues``, ``reference_thresholds``,
        ``n_recommended`` (leading-run count), ``total_passes`` (total count
        above threshold — for diagnostic comparison), ``semantics``.
    """
    n, p = data.shape
    R = _corr_safe(data)
    eig = _sorted_eigvals_desc(R)

    refs = np.zeros_like(eig)
    mp_upper = (1 + np.sqrt(p / n)) ** 2  # Marchenko–Pastur upper edge.
    cumulative = 0.0
    for k in range(len(eig)):
        # Variance remaining for factors k, k+1, ... after accounting for
        # eigenvalues 0..k-1.
        remaining = max(p - cumulative, 0.0)
        ref = mp_upper * remaining / max(p - k, 1)
        refs[k] = max(ref, 1.0)
        cumulative += eig[k]

    passes = eig > refs
    if passes.all():
        leading_run = len(eig)
    elif not passes.any():
        leading_run = 0
    else:
        leading_run = int(np.argmax(~passes))
    total_passes = int(np.sum(passes))

    return {
        "real_eigenvalues": eig.tolist(),
        "reference_thresholds": refs.tolist(),
        "n_recommended": leading_run,
        "total_passes": total_passes,
        "semantics": "leading_run",
    }


# ═════════════════════════════════════════════════════════════════════════════
# ACCELERATION FACTOR / OBJECTIVE SCREE
# ═════════════════════════════════════════════════════════════════════════════


def acceleration_factor(
    eigenvalues: np.ndarray,
    *,
    k_max: int | None = None,
) -> dict:
    """Objective scree test (Raîche, Roipel, Blais 2006).

    The second difference of the eigenvalue sequence picks out the scree
    elbow: AF(k) = λ_{k-1} − 2·λ_k + λ_{k+1}. Retain the k at the maximum
    AF — i.e., the eigenvalue just *before* the biggest drop-off in
    acceleration. Requires at least three eigenvalues.

    Note on wide psychometric data: if the first eigenvalue is much larger
    than the rest, λ_1 − 2λ_2 + λ_3 dominates AF and the method returns
    k=1 regardless of any real elbow further down the scree. Use ``k_max``
    to restrict attention to the first ``k_max`` eigenvalues — this
    mirrors the visual scree test, which also ignores the long rubble
    tail.

    Args:
        eigenvalues: Descending eigenvalues of the correlation matrix.
        k_max: Optional upper bound on the eigenvalues scanned. ``None`` =
            use every eigenvalue.

    Returns:
        Dict with ``af`` (length k_used−2, entry i refers to k=i+1),
        ``n_recommended`` (the k that maximizes AF), ``af_max``,
        ``k_max_used``, ``semantics``.
    """
    eig_full = np.asarray(eigenvalues, dtype=float).ravel()
    p_full = len(eig_full)
    k_used = p_full if k_max is None else min(int(k_max), p_full)
    eig = eig_full[:k_used]
    if len(eig) < 3:
        return {
            "af": [], "n_recommended": 0, "af_max": 0.0,
            "k_max_used": k_used, "semantics": "argmax_second_diff",
        }
    af = eig[:-2] - 2 * eig[1:-1] + eig[2:]  # length len(eig) - 2
    # af[i] corresponds to evaluating the second difference centered at
    # index i+1, i.e. comparing (λ_i, λ_{i+1}, λ_{i+2}). The elbow is at
    # λ_{i+1}, so the recommended number of factors to retain (those *before*
    # the elbow) is i+1. Take argmax over non-negative AF; treat ties and
    # negative maxima as "no elbow → retain 0".
    if np.max(af) <= 0:
        return {
            "af": af.tolist(), "n_recommended": 0,
            "af_max": float(np.max(af)), "k_max_used": k_used,
            "semantics": "argmax_second_diff",
        }
    k_elbow = int(np.argmax(af)) + 1
    return {
        "af": af.tolist(),
        "n_recommended": k_elbow,
        "af_max": float(np.max(af)),
        "k_max_used": k_used,
        "semantics": "argmax_second_diff",
    }


# ═════════════════════════════════════════════════════════════════════════════
# KAISER–GUTTMAN
# ═════════════════════════════════════════════════════════════════════════════


def kaiser_guttman(eigenvalues: np.ndarray) -> dict:
    """Classic eigenvalue > 1 rule on correlation-matrix eigenvalues.

    Included for transparency/reference only — known to over-retain on
    wide data. Downstream consumers should treat ``is_reference_only`` as
    a hint to de-emphasise this method in aggregated summaries.
    """
    eig = np.asarray(eigenvalues, dtype=float).ravel()
    n = int(np.sum(eig > 1.0))
    return {
        "n_recommended": n,
        "real_eigenvalues": eig.tolist(),
        "semantics": "total_count_above_threshold",
        "is_reference_only": True,
    }


# ═════════════════════════════════════════════════════════════════════════════
# OPTIMAL COORDINATES (Raîche, Walentin, Magis & Blais 2013)
# ═════════════════════════════════════════════════════════════════════════════


def optimal_coordinates(
    eigenvalues: np.ndarray,
    *,
    k_max: int | None = None,
) -> dict:
    """Optimal Coordinates (OC) scree test.

    For each k in [1, p−2], fit a least-squares line to the *tail* scree
    points (k+1, λ_{k+1}), (k+2, λ_{k+2}), ..., (p, λ_p) and use it to
    predict λ̂_k at x=k. Factor k "passes" OC if the observed λ_k exceeds
    the tail-forecast λ̂_k — i.e., λ_k stands above the extrapolated trend
    of the noise tail. Retain the leading run of consecutive k's that pass
    (stop at the first k that falls at or below its predicted line).

    This is a global scree-shape criterion — it looks at deviation from the
    entire right-tail trend rather than a single local slope. Differs from
    ``acceleration_factor`` (pure local 2nd difference) and from
    ``scree_elbow`` (perpendicular distance to first-last chord).

    Note: OC is known to be **liberal** — it over-extracts when noise
    eigenvalues have non-trivial variance around the tail trend, because
    individual noise eigenvalues can sit slightly above the predicted line
    and extend the "leading run" beyond the true dimensionality. Interpret
    as an upper-bound alongside the more conservative Horn / MAP estimates.

    Reference: Raîche, Walentin, Magis & Blais (2013), "Non-graphical
    solutions for Cattell's scree test", Methodology 9(1): 23-29.

    Args:
        eigenvalues: Descending eigenvalues of the correlation matrix.
        k_max: Optional upper bound on the number of eigenvalues used. With
            psychometric data (many items), the long rubble tail biases the
            least-squares line toward very small predictions, so every early
            eigenvalue trivially "passes" — OC then reports k close to ``p``.
            Truncating to the first ``k_max`` eigenvalues forces the line to
            fit the near-tail, which mirrors how analysts do the visual test.
            Default ``None`` = use all ``p`` eigenvalues.

    Returns:
        ``{"n_recommended": k, "predicted_eigenvalues": list,
        "real_eigenvalues": list, "passes": list[bool], "k_max_used": int}``.
    """
    eig_full = np.asarray(eigenvalues, dtype=float).ravel()
    p_full = len(eig_full)
    if k_max is None and p_full > 40:
        _log.warning(
            "optimal_coordinates: no k_max given on %d-eigenvalue scree — "
            "long rubble tail will bias the tail-line flat and OC will "
            "over-extract. Pass k_max to restrict to the informative head.",
            p_full,
        )
    k_used = p_full if k_max is None else min(int(k_max), p_full)
    eig = eig_full[:k_used]
    p = len(eig)
    if p < 3:
        return {
            "n_recommended": 0,
            "predicted_eigenvalues": [],
            "real_eigenvalues": eig_full.tolist(),
            "passes": [],
            "k_max_used": k_used,
            "semantics": "leading_run",
        }
    predicted = np.full(p, np.nan, dtype=float)
    passes = [False] * p
    for k in range(1, p - 1):
        tail_x = np.arange(k + 1, p + 1, dtype=float)
        tail_y = eig[k:]
        if len(tail_x) < 2:
            break
        b, a = np.polyfit(tail_x, tail_y, 1)
        lam_hat_k = a + b * float(k)
        predicted[k - 1] = lam_hat_k
        passes[k - 1] = bool(eig[k - 1] > lam_hat_k)

    n = 0
    for passed in passes:
        if passed:
            n += 1
        else:
            break
    return {
        "n_recommended": int(n),
        "predicted_eigenvalues": [
            float(v) if not np.isnan(v) else None for v in predicted
        ],
        "real_eigenvalues": eig_full.tolist(),
        "passes": passes,
        "k_max_used": k_used,
        "semantics": "leading_run",
    }


# ═════════════════════════════════════════════════════════════════════════════
# SCREE ELBOW — MAX PERPENDICULAR DISTANCE TO FIRST-LAST LINE (Kneedle-style)
# ═════════════════════════════════════════════════════════════════════════════


def scree_elbow(
    eigenvalues: np.ndarray,
    *,
    k_max: int | None = None,
) -> dict:
    """Programmatic scree elbow via perpendicular distance to the first-last line.

    Draws a chord between (1, λ_1) and (p, λ_p) in scree coordinates, then
    picks k at the eigenvalue that lies furthest below that chord
    (maximum perpendicular distance). Returns ``k_elbow`` as the number of
    factors to retain (i.e., the elbow index).

    Differs from ``acceleration_factor``: this is a global curvature measure
    (deviation from the linear trend over the whole scree), while AF is a
    purely local second-difference at each point. The two often agree within
    ±1 but disagree when the scree has a smooth curve vs. a sharp kink.

    Args:
        eigenvalues: Descending eigenvalues of the correlation matrix.
        k_max: Optional upper bound on the number of eigenvalues used to
            compute the chord. On wide psychometric data (many items) the
            trace-preservation of the correlation matrix forces a long
            rubble floor around 0.5-2.0; the chord from (1, λ_1) to
            (p, λ_p) then runs almost horizontally and max-perpendicular
            distance lands somewhere in the middle of the tail rather than
            at the real elbow. Truncating to the first ``k_max`` eigenvalues
            mirrors how analysts do the visual test. Default ``None`` = use
            all eigenvalues.

    Returns:
        ``{"n_recommended": k, "knee_index": k, "distances": list,
        "real_eigenvalues": list, "k_max_used": int}``.
    """
    eig_full = np.asarray(eigenvalues, dtype=float).ravel()
    p_full = len(eig_full)
    if k_max is None and p_full > 40:
        _log.warning(
            "scree_elbow: no k_max given on %d-eigenvalue scree — the "
            "first-to-last chord runs nearly flat and the knee will land "
            "in the rubble tail. Pass k_max to restrict to the informative "
            "head.",
            p_full,
        )
    k_used = p_full if k_max is None else min(int(k_max), p_full)
    eig = eig_full[:k_used]
    p = len(eig)
    # Need p >= 4 so that, after excluding the first and last points from
    # argmax, there are >=2 candidate knees to distinguish.
    if p < 4:
        return {
            "n_recommended": 0,
            "knee_index": 0,
            "distances": [],
            "real_eigenvalues": eig_full.tolist(),
            "k_max_used": k_used,
            "semantics": "cattell_retain_above_knee",
        }
    x = np.arange(1, p + 1, dtype=float)
    # Line from (1, λ_1) to (p, λ_p). Perpendicular distance from each point:
    # d_i = |(x_p - x_1)(y_1 - y_i) - (x_1 - x_i)(y_p - y_1)| / L
    x1, y1 = x[0], eig[0]
    xp, yp = x[-1], eig[-1]
    num = np.abs((xp - x1) * (y1 - eig) - (x1 - x) * (yp - y1))
    L = float(np.hypot(xp - x1, yp - y1))
    distances = num / (L if L > 0 else 1.0)
    # Restrict to interior points (elbow can't be the first or last eigenvalue).
    interior = distances.copy()
    interior[0] = -np.inf
    interior[-1] = -np.inf
    # Argmax gives the knee (1-indexed) — the point most below the chord, which
    # sits at the top of the noise floor. By Cattell's convention we retain
    # the factors *above* the elbow, so n_recommended = knee_index − 1.
    knee_idx_1 = int(np.argmax(interior)) + 1
    n_recommended = max(0, knee_idx_1 - 1)
    return {
        "n_recommended": n_recommended,
        "knee_index": knee_idx_1,
        "distances": distances.tolist(),
        "real_eigenvalues": eig_full.tolist(),
        "k_max_used": k_used,
        "semantics": "cattell_retain_above_knee",
    }


# ═════════════════════════════════════════════════════════════════════════════
# CV RECONSTRUCTION K — HELD-OUT PERSONAS, GAUSSIAN NLL
# ═════════════════════════════════════════════════════════════════════════════


def _tqdm(iterable, **kwargs):
    try:
        from tqdm.auto import tqdm  # noqa: F401
        return tqdm(iterable, **kwargs)
    except ImportError:
        return iterable


def _gaussian_nll_per_row(
    x: np.ndarray,
    loadings: np.ndarray,
    uniquenesses: np.ndarray,
) -> np.ndarray:
    """Per-row Gaussian NLL under Σ = ΛΛ' + Ψ (Ψ diagonal).

    Uses the matrix determinant lemma + Woodbury identity to avoid forming
    the full Σ⁻¹ / |Σ| for large p.
    """
    psi = np.maximum(uniquenesses, 1e-6)                     # [p]
    Lt_psi = loadings.T / psi[None, :]                        # [k, p]
    inner = np.eye(loadings.shape[1]) + Lt_psi @ loadings     # [k, k]
    sign, logdet_inner = np.linalg.slogdet(inner)
    if sign <= 0:
        # Degenerate inner; fall back to eigendecomposition on Σ directly.
        Sigma = loadings @ loadings.T + np.diag(psi)
        sign_sig, logdet_sig = np.linalg.slogdet(Sigma)
        if sign_sig <= 0:
            return np.full(x.shape[0], np.inf)
        logdet_sig = float(logdet_sig)
        Sigma_inv = np.linalg.inv(Sigma)
        quad = np.einsum("ni,ij,nj->n", x, Sigma_inv, x)
        p = x.shape[1]
        return 0.5 * (logdet_sig + quad + p * np.log(2 * np.pi))

    # Σ⁻¹ x = Ψ⁻¹ x − Ψ⁻¹ Λ inner⁻¹ Λᵀ Ψ⁻¹ x.
    inner_inv = np.linalg.inv(inner)
    psi_inv_x = x / psi[None, :]                              # [n, p]
    rhs = Lt_psi @ x.T                                        # [k, n]
    sigma_inv_x = psi_inv_x.T - (loadings / psi[:, None]) @ (inner_inv @ rhs)  # [p, n]
    quad = np.einsum("ni,in->n", x, sigma_inv_x)              # [n]

    # log|Σ| = log|inner| + Σ log ψ_j.
    logdet_sigma = logdet_inner + float(np.sum(np.log(psi)))
    p = x.shape[1]
    return 0.5 * (logdet_sigma + quad + p * np.log(2 * np.pi))


def cv_reconstruction_k(
    data: np.ndarray,
    *,
    k_max: int = 15,
    n_folds: int = 5,
    fa_method: str = "principal",
    rotation: str | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """K-fold held-out-persona Gaussian NLL across k.

    For each k ∈ 1..k_max and each of ``n_folds`` persona splits:
        * Fit FA on the training personas (standardized by train mean/std).
        * Form Σ(k) = Λ Λᵀ + diag(Ψ) at unit variance.
        * Score the held-out personas' rows (same standardization) under the
          MVN Σ(k).
    Average NLL per persona across folds → mean NLL curve.

    Rotation is irrelevant for NLL (Σ is rotation-invariant), so by default
    we skip rotation to keep the per-k fit cheap.

    Caveat — MVN misspecification: the items are ordinal/Likert, not
    Gaussian. This biases the NLL curve, typically toward over-penalising
    higher k (the ΛΛ' + Ψ structure can't perfectly capture discrete
    plateaus). The relative ordering across k is still informative; treat
    the chosen k as a lower bound for continuous-FA purposes. For a fully
    principled fix, switch to polychoric correlations + an ordinal FA
    extraction (not currently wired).

    Args:
        data: [n_samples, n_vars] response matrix.
        k_max: Maximum k to evaluate (k ranges 1..k_max).
        n_folds: Persona folds.
        fa_method: Extraction — "principal", "minres", or "ml".
        rotation: Rotation or None (default). FA likelihoods are
            rotation-invariant; rotating costs time without changing NLL.
        seed: RNG seed.
        verbose: Emit tqdm progress bars when tqdm is installed.

    Returns:
        Dict with:
            mean_nll_per_k: [k_max] mean per-row NLL at each k (lower = better).
            sem_nll_per_k: standard error across folds.
            nll_matrix: [n_folds, k_max] raw per-fold NLL means.
            n_recommended: argmin mean_nll_per_k + 1 (k space is 1-indexed).
            n_recommended_1se: smallest k within 1 SE of the minimum (one-SE
                rule favoring simpler models).
            k_range: list [1..k_max].
    """
    rng = np.random.default_rng(seed)
    n_samples, n_vars = data.shape
    k_max_requested = int(k_max)
    k_max = min(k_max, n_vars - 1, n_samples - n_samples // n_folds - 1)
    if k_max < k_max_requested:
        _log.warning(
            "cv_reconstruction_k: clamping k_max from %d to %d (n_vars=%d, "
            "n_samples=%d, n_folds=%d)",
            k_max_requested, k_max, n_vars, n_samples, n_folds,
        )
    if k_max < 1:
        return {
            "mean_nll_per_k": [], "sem_nll_per_k": [], "nll_matrix": [],
            "n_recommended": 0, "n_recommended_1se": 0, "k_range": [],
            "semantics": "argmin_nll",
        }

    order = rng.permutation(n_samples)
    fold_sizes = np.full(n_folds, n_samples // n_folds, dtype=int)
    fold_sizes[: n_samples % n_folds] += 1
    fold_bounds = np.concatenate([[0], np.cumsum(fold_sizes)])

    k_range = list(range(1, k_max + 1))
    nll_matrix = np.full((n_folds, k_max), np.nan)

    outer_iter = _tqdm(
        range(n_folds), desc="CV-k folds", disable=not verbose,
    )
    for fold in outer_iter:
        test_idx = order[fold_bounds[fold]: fold_bounds[fold + 1]]
        train_idx = np.concatenate([
            order[: fold_bounds[fold]], order[fold_bounds[fold + 1]:],
        ])
        train = data[train_idx]
        test = data[test_idx]

        # Standardize test/train by train mean+std: FactorAnalyzer computes
        # the correlation matrix internally, so returned loadings sit on a
        # unit-variance scale. Scoring Σ_model = ΛΛ' + Ψ against non-
        # standardized data is a scale mismatch that makes the NLL curve
        # uninterpretable. Standardizing matches test data to the implicit
        # model scale.
        train_mean = train.mean(axis=0)
        train_std = train.std(axis=0, ddof=0)
        train_std = np.where(train_std > 0, train_std, 1.0)
        train_std_arr = train_std  # alias for clarity
        train_z = (train - train_mean) / train_std_arr
        test_z = (test - train_mean) / train_std_arr

        inner_iter = _tqdm(
            k_range, desc=f"  fold {fold}", disable=not verbose, leave=False,
        )
        for j, k in enumerate(inner_iter):
            try:
                fa = run_factor_analysis(
                    train_z, n_factors=k, method=fa_method, rotation=rotation,
                )
            except Exception as exc:
                _log.warning("CV-k: FA failed at fold=%d k=%d: %s", fold, k, exc)
                continue
            loadings = fa["loadings"]
            uniq = np.maximum(1.0 - fa["communalities"], 1e-6)
            nll_rows = _gaussian_nll_per_row(test_z, loadings, uniq)
            nll_matrix[fold, j] = float(np.mean(nll_rows))

    mean_nll = np.nanmean(nll_matrix, axis=0)
    sem_nll = np.nanstd(nll_matrix, axis=0, ddof=1) / np.sqrt(
        np.maximum(np.sum(~np.isnan(nll_matrix), axis=0), 1)
    )

    if np.all(np.isnan(mean_nll)):
        return {
            "mean_nll_per_k": mean_nll.tolist(), "sem_nll_per_k": sem_nll.tolist(),
            "nll_matrix": nll_matrix.tolist(), "n_recommended": 0,
            "n_recommended_1se": 0, "k_range": k_range,
            "semantics": "argmin_nll", "saturated": False,
        }

    k_min = int(np.nanargmin(mean_nll))
    n_recommended = k_min + 1
    threshold = mean_nll[k_min] + sem_nll[k_min]
    # One-SE rule: smallest k whose mean NLL is within 1 SE of the min.
    within = np.where(mean_nll <= threshold)[0]
    n_recommended_1se = int(within.min()) + 1 if within.size else n_recommended

    saturated = bool(n_recommended == k_max)

    return {
        "mean_nll_per_k": mean_nll.tolist(),
        "sem_nll_per_k": sem_nll.tolist(),
        "nll_matrix": nll_matrix.tolist(),
        "n_recommended": n_recommended,
        "n_recommended_1se": n_recommended_1se,
        "k_range": k_range,
        "n_folds": int(n_folds),
        "k_max_used": int(k_max),
        "k_max_requested": int(k_max_requested),
        "semantics": "argmin_nll",
        "saturated": saturated,
    }


# ═════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════


_DEFAULT_METHODS = (
    "parallel", "map", "ekc", "acceleration", "kaiser", "cv_reconstruction",
    "optimal_coordinates", "scree_elbow",
)


def suggest_n_factors(
    data: np.ndarray,
    *,
    methods: tuple[str, ...] = _DEFAULT_METHODS,
    k_max: int = 15,
    parallel_n_iterations: int = 100,
    cv_n_folds: int = 5,
    fa_method: str = "principal",
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Run a suite of n-factors methods and return k recommendations + diagnostics.

    Note on cross-method semantics: each method's ``n_recommended`` uses its
    own convention (see each method's ``semantics`` key in the returned
    dict). In particular:

        * parallel / ekc / optimal_coordinates → ``leading_run`` (first k
          where the null threshold is crossed, conservative).
        * scree_elbow → ``cattell_retain_above_knee`` (knee − 1).
        * acceleration → ``argmax_second_diff``.
        * kaiser → ``total_count_above_threshold`` (reference-only; tends
          to over-extract).
        * map / cv_reconstruction → ``argmin`` of their respective curves;
          may saturate at k_max (see the ``saturated`` flag).

    Args:
        data: [n_samples, n_vars] response matrix. Should already be cleaned
            (zero-variance columns removed) by the caller.
        methods: Which methods to run. Any subset of
            ("parallel", "map", "ekc", "acceleration", "kaiser",
             "cv_reconstruction", "optimal_coordinates", "scree_elbow").
        k_max: Maximum k for MAP, CV, OC, scree_elbow, and acceleration.
            Parallel analysis runs over full p (its null correctly handles
            the whole scree).
        parallel_n_iterations: Iterations for Horn's permutation null.
        cv_n_folds: Folds for the CV method.
        fa_method: Extraction passed to the CV method (PAF by default).
        seed: RNG seed.
        verbose: Print per-method recommendations.

    Returns:
        Dict keyed by method name, each value is the method's own return dict.
        Also includes ``summary``: a dict mapping method → n_recommended,
        and ``summary_meta``: a dict mapping method → {semantics, saturated,
        is_reference_only} for downstream aggregation.
    """
    from src_dev.factor_analysis.parallel_analysis import parallel_analysis

    results: dict[str, dict] = {}

    if "parallel" in methods:
        if verbose:
            print("  [n_factors] Parallel analysis (permutation null)...")
        results["parallel"] = parallel_analysis(
            data, n_iterations=parallel_n_iterations,
            random_state=seed, method="permutation",
        )

    if "map" in methods:
        if verbose:
            print("  [n_factors] Velicer's MAP...")
        results["map"] = velicer_map(data, k_max=k_max)

    if "ekc" in methods:
        if verbose:
            print("  [n_factors] Empirical Kaiser Criterion...")
        results["ekc"] = empirical_kaiser_criterion(data)

    # Acceleration, Kaiser, OC, and scree_elbow all need the real eigenvalues
    # — compute once.
    _scree_methods = {
        "acceleration", "kaiser", "optimal_coordinates", "scree_elbow",
    }
    if _scree_methods & set(methods):
        R = _corr_safe(data)
        eig = _sorted_eigvals_desc(R)
        if "acceleration" in methods:
            if verbose:
                print(f"  [n_factors] Acceleration factor (k_max={k_max})...")
            results["acceleration"] = acceleration_factor(eig, k_max=k_max)
        if "kaiser" in methods:
            if verbose:
                print("  [n_factors] Kaiser–Guttman...")
            results["kaiser"] = kaiser_guttman(eig)
        if "optimal_coordinates" in methods:
            if verbose:
                print(f"  [n_factors] Optimal Coordinates (Raîche 2013, k_max={k_max})...")
            results["optimal_coordinates"] = optimal_coordinates(eig, k_max=k_max)
        if "scree_elbow" in methods:
            if verbose:
                print(f"  [n_factors] Scree elbow (max perp distance, k_max={k_max})...")
            results["scree_elbow"] = scree_elbow(eig, k_max=k_max)

    if "cv_reconstruction" in methods:
        if verbose:
            print(
                f"  [n_factors] CV reconstruction "
                f"(K={cv_n_folds} folds, k=1..{k_max})..."
            )
        results["cv_reconstruction"] = cv_reconstruction_k(
            data, k_max=k_max, n_folds=cv_n_folds,
            fa_method=fa_method, rotation=None, seed=seed + 1,
            verbose=verbose,
        )

    summary = {m: r.get("n_recommended", None) for m, r in results.items()}
    if "cv_reconstruction" in results:
        summary["cv_reconstruction_1se"] = results["cv_reconstruction"].get(
            "n_recommended_1se", None,
        )
    if "map" in results:
        summary["map_fm4"] = results["map"].get("n_recommended_fm4", None)
    results["summary"] = summary

    summary_meta: dict[str, dict] = {}
    for method, r in results.items():
        if method in ("summary", "summary_meta"):
            continue
        summary_meta[method] = {
            "semantics": r.get("semantics"),
            "saturated": r.get("saturated", False),
            "is_reference_only": r.get("is_reference_only", False),
        }
    results["summary_meta"] = summary_meta

    if verbose:
        print("  [n_factors] Summary:")
        for method, k in summary.items():
            meta = summary_meta.get(method, {})
            flags = []
            if meta.get("saturated"):
                flags.append("SATURATED")
            if meta.get("is_reference_only"):
                flags.append("reference-only")
            flag_str = f"  [{', '.join(flags)}]" if flags else ""
            print(f"    {method:>28}: k = {k}{flag_str}")

    return results


# ═════════════════════════════════════════════════════════════════════════════
# PLOTTING
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


def plot_n_factors_comparison(
    result: dict,
    save_path: Path,
    *,
    plt: Any | None = None,
    title_suffix: str = "",
) -> Path | None:
    """Four-panel comparison plot: scree + PA, MAP, EKC, CV-NLL.

    Panel A: Scree with Horn's permutation threshold, EKC threshold, and a
             vertical line at each method's recommended k.
    Panel B: Velicer's MAP curve (fm and fm4).
    Panel C: Acceleration factor (second difference of eigenvalues).
    Panel D: CV-reconstruction NLL vs. k with 1-SE band.

    Args:
        result: Output of ``suggest_n_factors``.
        save_path: PNG output path.
        plt: Optional matplotlib.pyplot module.
        title_suffix: Extra text appended to the super-title (e.g. the FA
            variant key).

    Returns:
        Path to saved file or None if matplotlib unavailable.
    """
    plt_mod = _resolve_plt(plt)
    if plt_mod is None:
        return None

    summary = result.get("summary", {})
    fig, axes = plt_mod.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    ax_scree, ax_map, ax_af, ax_cv = axes.flat

    # ── Panel A: Scree + PA / EKC / Kaiser lines ─────────────────────────
    pa = result.get("parallel")
    ekc = result.get("ekc")
    kg = result.get("kaiser")
    eig = None
    if pa is not None:
        eig = np.array(pa["real_eigenvalues"])
    elif ekc is not None:
        eig = np.array(ekc["real_eigenvalues"])
    elif kg is not None:
        eig = np.array(kg["real_eigenvalues"])

    if eig is not None:
        # Extend the x-range so that every method's recommended k is visible
        # as a vertical line, even if it sits past the usual 30-eigenvalue
        # scree clip. Leave a small pad past the largest recommended k.
        recommended_ks = [
            int(v) for v in summary.values()
            if isinstance(v, (int, np.integer)) and v is not None and v >= 1
        ]
        max_rec_k = max(recommended_ks, default=0)
        desired = max(30, max_rec_k + 2)
        n_show = min(len(eig), desired)
        x = np.arange(1, n_show + 1)
        ax_scree.plot(x, eig[:n_show], "o-", color="#1e40af", linewidth=2,
                      markersize=5, label="Real eigenvalues", zorder=5)
        if pa is not None:
            ax_scree.plot(
                x, np.array(pa["random_threshold"])[:n_show],
                "--", color="#dc2626", linewidth=1.2,
                label=f"Horn 95pct (k={summary.get('parallel', '?')})",
                zorder=4,
            )
        if ekc is not None:
            ax_scree.plot(
                x, np.array(ekc["reference_thresholds"])[:n_show],
                ":", color="#7c3aed", linewidth=1.2,
                label=f"EKC threshold (k={summary.get('ekc', '?')})",
                zorder=3,
            )
        ax_scree.axhline(1.0, color="#9ca3af", linewidth=0.8, linestyle="-.",
                         label=f"Kaiser (k={summary.get('kaiser', '?')})",
                         zorder=2)

        # Optimal Coordinates: overlay the predicted line-forecast for each k.
        oc = result.get("optimal_coordinates")
        if oc is not None and len(oc.get("predicted_eigenvalues", [])) > 0:
            pred_full = np.array([
                np.nan if v is None else float(v)
                for v in oc["predicted_eigenvalues"]
            ], dtype=float)
            # Plot at k-indices 1..p-2 where prediction is defined. pred_full
            # is clipped to OC's k_max_used (can be < n_show when the caller
            # passes k_max < p-2), so trim x to match or the mask rank will
            # diverge from x's rank on the indexing step below.
            pred_show = pred_full[:n_show]
            x_oc = x[: len(pred_show)]
            mask = ~np.isnan(pred_show)
            if mask.any():
                ax_scree.plot(
                    x_oc[mask], pred_show[mask],
                    ":", color="#be185d", linewidth=1.2,
                    label=f"OC predicted (k={summary.get('optimal_coordinates', '?')})",
                    zorder=3,
                )

        # Scree elbow: mark the knee (where the drop happens) with a hollow
        # circle; the recommended retain-count is knee − 1.
        se = result.get("scree_elbow")
        if se is not None and se.get("knee_index"):
            knee = int(se["knee_index"])
            k_se = int(summary.get("scree_elbow") or 0)
            if 1 <= knee <= n_show:
                ax_scree.plot(
                    [knee], [eig[knee - 1]],
                    marker="o", markersize=12, markerfacecolor="none",
                    markeredgecolor="#0f766e", markeredgewidth=2,
                    linestyle="none",
                    label=f"Scree elbow knee (retain k={k_se})",
                    zorder=6,
                )

        # Vertical lines at each method's k (clipped to axis range).
        color_by_method = {
            "parallel": "#dc2626",
            "ekc": "#7c3aed",
            "kaiser": "#9ca3af",
            "map": "#059669",
            "acceleration": "#ea580c",
            "cv_reconstruction": "#2563eb",
            "cv_reconstruction_1se": "#60a5fa",
            "optimal_coordinates": "#be185d",
            "scree_elbow": "#0f766e",
        }
        for method, col in color_by_method.items():
            k = summary.get(method)
            if not k or k < 1 or k > n_show:
                continue
            ax_scree.axvline(k, color=col, alpha=0.25, linewidth=1.2, zorder=1)
        ax_scree.set_xlabel("Component index")
        ax_scree.set_ylabel("Eigenvalue")
        ax_scree.set_title("Scree + thresholds", fontsize=11, fontweight="bold")
        ax_scree.legend(fontsize=8, loc="best")
        ax_scree.grid(alpha=0.3)

    # ── Panel B: MAP ────────────────────────────────────────────────────
    mp = result.get("map")
    if mp is not None and len(mp["fm"]) > 0:
        fm = np.array(mp["fm"])
        fm4 = np.array(mp["fm4"])
        kx = np.arange(len(fm))
        ax_map.plot(kx, fm, "o-", color="#059669", label=f"fm (k={summary.get('map', '?')})")
        ax_map.plot(kx, fm4, "s--", color="#16a34a",
                    label=f"fm4 (k={summary.get('map_fm4', '?')})")
        ax_map.axvline(summary.get("map", 0), color="#059669", alpha=0.3)
        ax_map.set_xlabel("k factors partialed out")
        ax_map.set_ylabel("Mean squared partial correlation")
        ax_map.set_title("Velicer's MAP", fontsize=11, fontweight="bold")
        ax_map.legend(fontsize=9)
        ax_map.grid(alpha=0.3)

    # ── Panel C: Acceleration factor ─────────────────────────────────────
    af = result.get("acceleration")
    if af is not None and len(af.get("af", [])) > 0:
        af_arr = np.array(af["af"])
        kx = np.arange(1, 1 + len(af_arr))
        ax_af.bar(kx, af_arr, color="#ea580c", edgecolor="white", zorder=3)
        ax_af.axvline(summary.get("acceleration", 0), color="#ea580c", alpha=0.3)
        ax_af.set_xlabel("k (number of factors retained)")
        ax_af.set_ylabel("Acceleration (λ_{k-1} − 2λ_k + λ_{k+1})")
        ax_af.set_title(
            f"Acceleration factor (k={summary.get('acceleration', '?')})",
            fontsize=11, fontweight="bold",
        )
        ax_af.grid(axis="y", alpha=0.3)

    # ── Panel D: CV NLL ──────────────────────────────────────────────────
    cv = result.get("cv_reconstruction")
    if cv is not None and len(cv.get("mean_nll_per_k", [])) > 0:
        mean = np.array(cv["mean_nll_per_k"])
        sem = np.array(cv["sem_nll_per_k"])
        kx = np.array(cv["k_range"])
        ax_cv.plot(kx, mean, "o-", color="#2563eb", linewidth=2, zorder=3)
        ax_cv.fill_between(kx, mean - sem, mean + sem, color="#2563eb",
                           alpha=0.2, zorder=2)
        k_rec = summary.get("cv_reconstruction")
        k_1se = summary.get("cv_reconstruction_1se")
        if k_rec is not None and k_rec >= 1:
            ax_cv.axvline(k_rec, color="#2563eb", linestyle="-", alpha=0.4,
                          label=f"argmin (k={k_rec})")
        if k_1se is not None and k_1se >= 1 and k_1se != k_rec:
            ax_cv.axvline(k_1se, color="#60a5fa", linestyle="--", alpha=0.6,
                          label=f"1-SE (k={k_1se})")
        ax_cv.set_xlabel("k (number of factors)")
        ax_cv.set_ylabel("Mean held-out NLL per persona")
        ax_cv.set_title(
            f"CV reconstruction ({cv.get('n_folds', '?')} folds)",
            fontsize=11, fontweight="bold",
        )
        ax_cv.legend(fontsize=9)
        ax_cv.grid(alpha=0.3)

    # ── Suptitle table of recommendations ────────────────────────────────
    rec_lines = [f"{m}: k={k}" for m, k in summary.items() if k is not None]
    # Wrap the per-method recs over multiple lines when many methods are run,
    # otherwise the suptitle runs off the figure.
    per_row = 5
    wrapped = [
        "  |  ".join(rec_lines[i:i + per_row])
        for i in range(0, len(rec_lines), per_row)
    ]
    rec_str = "\n".join(wrapped)
    fig.suptitle(
        f"Number-of-factors suggestions{(' — ' + title_suffix) if title_suffix else ''}\n{rec_str}",
        fontsize=11, fontweight="bold",
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt_mod.close(fig)
    return save_path
