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
    """Correlation matrix that tolerates zero-variance columns (→ identity row)."""
    std = data.std(axis=0, ddof=0)
    if np.any(std == 0):
        # Replace zero-variance columns with a tiny ridge so np.corrcoef doesn't NaN.
        data = data + np.where(std == 0, 1e-12, 0.0) * np.random.default_rng(0).standard_normal(data.shape)
    return np.corrcoef(data, rowvar=False)


def _sorted_eigvals_desc(mat: np.ndarray) -> np.ndarray:
    """Descending eigenvalues of a symmetric matrix."""
    return np.sort(np.linalg.eigvalsh(mat))[::-1]


# ═════════════════════════════════════════════════════════════════════════════
# VELICER'S MAP
# ═════════════════════════════════════════════════════════════════════════════


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
        k_max: Maximum k to test. Defaults to min(n_vars − 2, 30).

    Returns:
        Dict with keys ``fm``, ``fm4`` (both length k_max+1; fm[0] is the
        mean squared off-diagonal of R itself), ``n_recommended`` (argmin fm,
        revised MAP recommendation), and ``n_recommended_fm4`` (argmin fm4).
    """
    n_samples, n_vars = data.shape
    if k_max is None:
        k_max = min(n_vars - 2, 30)
    if k_max < 1:
        return {"fm": [], "fm4": [], "n_recommended": 0, "n_recommended_fm4": 0}

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
        return float(np.mean(mat[off_diag_mask] ** power) / 1.0)

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
        np.fill_diagonal(R_k, 0.0)  # ensure diagonal contributes nothing.
        fm_list.append(_mean_sq_offdiag(R_k + np.eye(n_vars), 2))  # add I back only for mask
        fm4_list.append(_mean_sq_offdiag(R_k + np.eye(n_vars), 4))

    fm_arr = np.array(fm_list, dtype=float)
    fm4_arr = np.array(fm4_list, dtype=float)

    # n_recommended is the k at minimum. fm[0] corresponds to k=0 (retain 0).
    # Typically MAP is reported as argmin on k >= 1 — retaining zero factors
    # isn't usable. Return argmin over k >= 1 when available, falling back to
    # 0 when the min really is at k=0 (data is already near-diagonal).
    def _argmin_recommendation(arr: np.ndarray) -> int:
        k_star = int(np.argmin(arr))
        return k_star  # 0-indexed over k values 0..len-1

    return {
        "fm": fm_arr.tolist(),
        "fm4": fm4_arr.tolist(),
        "n_recommended": _argmin_recommendation(fm_arr),
        "n_recommended_fm4": _argmin_recommendation(fm4_arr),
        "k_max_tested": len(fm_arr) - 1,
    }


# ═════════════════════════════════════════════════════════════════════════════
# EMPIRICAL KAISER CRITERION
# ═════════════════════════════════════════════════════════════════════════════


def empirical_kaiser_criterion(data: np.ndarray) -> dict:
    """Empirical Kaiser Criterion (Braeken & van Assen 2017).

    For each k, the reference eigenvalue λ_ref(k) under the null of no
    additional factors is derived from the Marchenko–Pastur distribution
    (asymptotic eigenvalue density of Wishart correlation matrices). The
    retained k is the largest index where:

        λ_real(k) > max( λ_ref(k),  1 )

    with λ_ref(k) = (1 + √(p / n))² · (p − Σ_{j<k} λ_real(j)) / (p − k + 1).
    This adapts for variance already accounted for at lower k, fixing the
    Kaiser rule's tendency to over-retain.

    Args:
        data: [n_samples, n_vars] response matrix.

    Returns:
        Dict with ``real_eigenvalues``, ``reference_thresholds``,
        ``n_recommended``.
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

    n_recommended = int(np.sum(eig > refs))

    return {
        "real_eigenvalues": eig.tolist(),
        "reference_thresholds": refs.tolist(),
        "n_recommended": n_recommended,
    }


# ═════════════════════════════════════════════════════════════════════════════
# ACCELERATION FACTOR / OBJECTIVE SCREE
# ═════════════════════════════════════════════════════════════════════════════


def acceleration_factor(eigenvalues: np.ndarray) -> dict:
    """Objective scree test (Raîche, Roipel, Blais 2006).

    The second difference of the eigenvalue sequence picks out the scree
    elbow: AF(k) = λ_{k-1} − 2·λ_k + λ_{k+1}. Retain the k at the maximum
    AF — i.e., the eigenvalue just *before* the biggest drop-off in
    acceleration. Requires at least three eigenvalues.

    Args:
        eigenvalues: Descending eigenvalues of the correlation matrix.

    Returns:
        Dict with ``af`` (length len(eig)−2, entry i refers to k=i+1),
        ``n_recommended`` (the k that maximizes AF), and ``af_max``.
    """
    eig = np.asarray(eigenvalues, dtype=float).ravel()
    if len(eig) < 3:
        return {"af": [], "n_recommended": 0, "af_max": 0.0}
    af = eig[:-2] - 2 * eig[1:-1] + eig[2:]  # length len(eig) - 2
    # af[i] corresponds to evaluating the second difference centered at
    # index i+1, i.e. comparing (λ_i, λ_{i+1}, λ_{i+2}). The elbow is at
    # λ_{i+1}, so the recommended number of factors to retain (those *before*
    # the elbow) is i+1. Take argmax over non-negative AF; treat ties and
    # negative maxima as "no elbow → retain 0".
    if np.max(af) <= 0:
        return {"af": af.tolist(), "n_recommended": 0, "af_max": float(np.max(af))}
    k_elbow = int(np.argmax(af)) + 1
    return {"af": af.tolist(), "n_recommended": k_elbow, "af_max": float(np.max(af))}


# ═════════════════════════════════════════════════════════════════════════════
# KAISER–GUTTMAN
# ═════════════════════════════════════════════════════════════════════════════


def kaiser_guttman(eigenvalues: np.ndarray) -> dict:
    """Classic eigenvalue > 1 rule on correlation-matrix eigenvalues."""
    eig = np.asarray(eigenvalues, dtype=float).ravel()
    n = int(np.sum(eig > 1.0))
    return {"n_recommended": n, "real_eigenvalues": eig.tolist()}


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
        * Fit FA on the training personas.
        * Form Σ(k) = Λ Λᵀ + diag(Ψ) on item-mean-centered data.
        * Score the held-out personas' rows under multivariate Gaussian NLL.
    Average NLL per persona across folds → mean NLL curve.

    Rotation is irrelevant for NLL (Σ is rotation-invariant), so by default
    we skip rotation to keep the per-k fit cheap.

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
    k_max = min(k_max, n_vars - 1, n_samples - n_samples // n_folds - 1)
    if k_max < 1:
        return {
            "mean_nll_per_k": [], "sem_nll_per_k": [], "nll_matrix": [],
            "n_recommended": 0, "n_recommended_1se": 0, "k_range": [],
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

        train_mean = train.mean(axis=0)
        train_c = train - train_mean
        test_c = test - train_mean

        inner_iter = _tqdm(
            k_range, desc=f"  fold {fold}", disable=not verbose, leave=False,
        )
        for j, k in enumerate(inner_iter):
            try:
                fa = run_factor_analysis(
                    train_c, n_factors=k, method=fa_method, rotation=rotation,
                )
            except Exception as exc:
                _log.warning("CV-k: FA failed at fold=%d k=%d: %s", fold, k, exc)
                continue
            loadings = fa["loadings"]
            uniq = np.maximum(1.0 - fa["communalities"], 1e-6)
            nll_rows = _gaussian_nll_per_row(test_c, loadings, uniq)
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
        }

    k_min = int(np.nanargmin(mean_nll))
    n_recommended = k_min + 1
    threshold = mean_nll[k_min] + sem_nll[k_min]
    # One-SE rule: smallest k whose mean NLL is within 1 SE of the min.
    within = np.where(mean_nll <= threshold)[0]
    n_recommended_1se = int(within.min()) + 1 if within.size else n_recommended

    return {
        "mean_nll_per_k": mean_nll.tolist(),
        "sem_nll_per_k": sem_nll.tolist(),
        "nll_matrix": nll_matrix.tolist(),
        "n_recommended": n_recommended,
        "n_recommended_1se": n_recommended_1se,
        "k_range": k_range,
        "n_folds": int(n_folds),
    }


# ═════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════


_DEFAULT_METHODS = (
    "parallel", "map", "ekc", "acceleration", "kaiser", "cv_reconstruction",
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

    Args:
        data: [n_samples, n_vars] response matrix. Should already be cleaned
            (zero-variance columns removed) by the caller.
        methods: Which methods to run. Any subset of
            ("parallel", "map", "ekc", "acceleration", "kaiser",
             "cv_reconstruction").
        k_max: Maximum k for MAP and CV. Parallel analysis runs over full p.
        parallel_n_iterations: Iterations for Horn's permutation null.
        cv_n_folds: Folds for the CV method.
        fa_method: Extraction passed to the CV method (PAF by default).
        seed: RNG seed.
        verbose: Print per-method recommendations.

    Returns:
        Dict keyed by method name, each value is the method's own return dict.
        Also includes ``summary``: a dict mapping method → n_recommended.
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

    # Acceleration and Kaiser both need the real eigenvalues — compute once.
    if "acceleration" in methods or "kaiser" in methods:
        R = _corr_safe(data)
        eig = _sorted_eigvals_desc(R)
        if "acceleration" in methods:
            if verbose:
                print("  [n_factors] Acceleration factor...")
            results["acceleration"] = acceleration_factor(eig)
        if "kaiser" in methods:
            if verbose:
                print("  [n_factors] Kaiser–Guttman...")
            results["kaiser"] = kaiser_guttman(eig)

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

    if verbose:
        print("  [n_factors] Summary:")
        for method, k in summary.items():
            print(f"    {method:>28}: k = {k}")

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
        n_show = min(len(eig), 30)
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
        # Vertical lines at each method's k (clipped to axis range).
        color_by_method = {
            "parallel": "#dc2626",
            "ekc": "#7c3aed",
            "kaiser": "#9ca3af",
            "map": "#059669",
            "acceleration": "#ea580c",
            "cv_reconstruction": "#2563eb",
            "cv_reconstruction_1se": "#60a5fa",
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
        if k_rec:
            ax_cv.axvline(k_rec, color="#2563eb", linestyle="-", alpha=0.4,
                          label=f"argmin (k={k_rec})")
        if k_1se and k_1se != k_rec:
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
    rec_str = "  |  ".join(rec_lines)
    fig.suptitle(
        f"Number-of-factors suggestions{(' — ' + title_suffix) if title_suffix else ''}\n{rec_str}",
        fontsize=12, fontweight="bold",
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt_mod.close(fig)
    return save_path
