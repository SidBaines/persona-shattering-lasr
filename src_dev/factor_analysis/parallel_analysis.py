"""Horn's parallel analysis for determining the number of factors to retain."""

from __future__ import annotations

import numpy as np


def parallel_analysis(
    data: np.ndarray,
    n_iterations: int = 100,
    percentile: float = 95.0,
    random_state: int = 42,
    method: str = "normal",
) -> dict:
    """Run Horn's parallel analysis.

    Compares eigenvalues of the real correlation matrix against eigenvalues
    from random data of the same shape to determine how many factors have
    more structure than noise.

    Args:
        data: Input matrix [n_samples, n_vars] (typically PCA-reduced).
        n_iterations: Number of random datasets to generate.
        percentile: Percentile of random eigenvalues to use as threshold.
        random_state: Random seed for reproducibility.
        method: How to generate the null reference data.
            "normal" — draw from a standard normal (classic Horn's method).
            "permutation" — independently permute each column of the input
            data, preserving marginal distributions. Recommended for ordinal
            data (e.g. Likert scales) where Gaussian reference eigenvalues
            can be systematically biased (Buja & Eyuboglu, 1992).

    Returns:
        Dict with keys:
            real_eigenvalues: Eigenvalues of the real correlation matrix (descending).
            random_threshold: Percentile threshold at each position.
            n_recommended: Number of factors where real > threshold.
            random_eigenvalues_all: All random eigenvalues [n_iterations, n_vars].
    """
    if method not in ("normal", "permutation"):
        raise ValueError(f"method must be 'normal' or 'permutation', got {method!r}")

    n_samples, n_vars = data.shape
    rng = np.random.default_rng(random_state)

    # Real eigenvalues from correlation matrix.
    corr = np.corrcoef(data, rowvar=False)
    real_eigenvalues = np.sort(np.linalg.eigvalsh(corr))[::-1]

    # Random eigenvalues.
    random_eigenvalues_all = np.zeros((n_iterations, n_vars), dtype=np.float64)
    for i in range(n_iterations):
        if method == "normal":
            random_data = rng.standard_normal((n_samples, n_vars))
        else:
            # Permutation: independently shuffle each column of the real data.
            random_data = data.copy()
            for j in range(n_vars):
                rng.shuffle(random_data[:, j])
        random_corr = np.corrcoef(random_data, rowvar=False)
        random_eigenvalues_all[i] = np.sort(np.linalg.eigvalsh(random_corr))[::-1]

    random_threshold = np.percentile(random_eigenvalues_all, percentile, axis=0)

    # Count factors where real exceeds threshold.
    n_recommended = int(np.sum(real_eigenvalues > random_threshold))

    print(f"Parallel analysis: {n_recommended} factors recommended "
          f"(real eigenvalue > {percentile}th percentile of random)")

    return {
        "real_eigenvalues": real_eigenvalues,
        "random_threshold": random_threshold,
        "n_recommended": n_recommended,
        "random_eigenvalues_all": random_eigenvalues_all,
    }
