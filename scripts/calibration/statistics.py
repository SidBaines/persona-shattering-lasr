"""Statistical helpers for calibration analyses."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def _as_float_array(values: list[float] | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float)


def zscore(values: list[float] | np.ndarray) -> tuple[np.ndarray, float, float, bool]:
    """Standardize values to z-scores.

    Args:
        values: Numeric values.

    Returns:
        Tuple of (z_values, mean, std, used_fallback) where used_fallback is True
        when standard deviation is zero and all z-values are set to 0.
    """
    arr = _as_float_array(values)
    finite_mask = np.isfinite(arr)
    finite = arr[finite_mask]
    out = np.full(arr.shape, np.nan, dtype=float)
    if finite.size == 0:
        return out, float("nan"), float("nan"), True
    mean = float(np.mean(finite))
    std = float(np.std(finite))
    if not np.isfinite(std) or std <= 0:
        out[finite_mask] = 0.0
        return out, mean, std, True
    out[finite_mask] = (arr[finite_mask] - mean) / std
    return out, mean, std, False


def rankdata_average(values: list[float] | np.ndarray) -> np.ndarray:
    """Compute average ranks with tie handling.

    Args:
        values: Numeric vector.

    Returns:
        Array of float ranks (1-indexed).
    """
    arr = _as_float_array(values)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=float)

    i = 0
    while i < len(arr):
        j = i
        while j + 1 < len(arr) and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def pearson_r(x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> float:
    """Compute Pearson correlation coefficient."""
    x_arr = _as_float_array(x)
    y_arr = _as_float_array(y)
    if x_arr.size < 2 or y_arr.size < 2:
        return float("nan")
    x_std = float(np.std(x_arr))
    y_std = float(np.std(y_arr))
    if x_std <= 0 or y_std <= 0:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def spearman_rho(x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> float:
    """Compute Spearman rank correlation coefficient."""
    x_arr = _as_float_array(x)
    y_arr = _as_float_array(y)
    if x_arr.size < 2 or y_arr.size < 2:
        return float("nan")
    return pearson_r(rankdata_average(x_arr), rankdata_average(y_arr))


def mae(y_true: list[float] | np.ndarray, y_pred: list[float] | np.ndarray) -> float:
    """Compute mean absolute error."""
    y_true_arr = _as_float_array(y_true)
    y_pred_arr = _as_float_array(y_pred)
    if y_true_arr.size == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true_arr - y_pred_arr)))


def rmse(y_true: list[float] | np.ndarray, y_pred: list[float] | np.ndarray) -> float:
    """Compute root mean squared error."""
    y_true_arr = _as_float_array(y_true)
    y_pred_arr = _as_float_array(y_pred)
    if y_true_arr.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))


def linear_calibration_fit(
    x: list[float] | np.ndarray,
    y: list[float] | np.ndarray,
) -> dict[str, float]:
    """Fit y = intercept + slope * x and compute R^2."""
    x_arr = _as_float_array(x)
    y_arr = _as_float_array(y)
    if x_arr.size < 2 or y_arr.size < 2:
        return {
            "slope": float("nan"),
            "intercept": float("nan"),
            "r2": float("nan"),
        }

    x_var = float(np.var(x_arr))
    if x_var <= 0:
        return {
            "slope": float("nan"),
            "intercept": float("nan"),
            "r2": float("nan"),
        }

    slope = float(np.cov(x_arr, y_arr, ddof=0)[0, 1] / x_var)
    intercept = float(np.mean(y_arr) - slope * np.mean(x_arr))

    y_hat = intercept + slope * x_arr
    residual = y_arr - y_hat
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    r2 = float("nan") if ss_tot <= 0 else float(1.0 - (ss_res / ss_tot))

    return {
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
    }


def validity_metrics(
    judge_z: list[float] | np.ndarray,
    gt_z: list[float] | np.ndarray,
) -> dict[str, float]:
    """Compute validity metrics over paired numeric vectors."""
    judge_arr = _as_float_array(judge_z)
    gt_arr = _as_float_array(gt_z)

    mask = np.isfinite(judge_arr) & np.isfinite(gt_arr)
    judge_arr = judge_arr[mask]
    gt_arr = gt_arr[mask]

    fit = linear_calibration_fit(judge_arr, gt_arr)
    out = {
        "n": float(judge_arr.size),
        "pearson_r": pearson_r(judge_arr, gt_arr),
        "spearman_rho": spearman_rho(judge_arr, gt_arr),
        "mae": mae(gt_arr, judge_arr),
        "rmse": rmse(gt_arr, judge_arr),
        "slope": fit["slope"],
        "intercept": fit["intercept"],
        "r2": fit["r2"],
    }
    return out


def _bootstrap_ci(
    values: list[float],
    alpha: float = 0.05,
) -> tuple[float, float] | None:
    finite = [x for x in values if np.isfinite(x)]
    if not finite:
        return None
    lower = float(np.quantile(finite, alpha / 2.0))
    upper = float(np.quantile(finite, 1.0 - (alpha / 2.0)))
    return lower, upper


def bootstrap_metric_cis(
    rows: np.ndarray,
    metric_fn: Callable[[np.ndarray], dict[str, float]],
    metric_names: list[str],
    *,
    n_samples: int,
    random_seed: int,
) -> dict[str, dict[str, float | int | None]]:
    """Bootstrap confidence intervals for metrics computed on row-wise data.

    Args:
        rows: 2D array where rows are independent units.
        metric_fn: Function returning named metric values.
        metric_names: Metric keys to summarize.
        n_samples: Number of bootstrap samples.
        random_seed: RNG seed.

    Returns:
        Mapping metric -> CI metadata.
    """
    rng = np.random.default_rng(random_seed)
    n_rows = rows.shape[0]

    sampled: dict[str, list[float]] = {name: [] for name in metric_names}
    for _ in range(n_samples):
        sample_idx = rng.integers(0, n_rows, size=n_rows)
        sample_rows = rows[sample_idx, :]
        metrics = metric_fn(sample_rows)
        for name in metric_names:
            sampled[name].append(float(metrics.get(name, float("nan"))))

    out: dict[str, dict[str, float | int | None]] = {}
    for name in metric_names:
        ci = _bootstrap_ci(sampled[name])
        if ci is None:
            out[name] = {
                "ci_low": None,
                "ci_high": None,
                "bootstrap_valid": 0,
                "bootstrap_samples": n_samples,
            }
        else:
            valid = sum(int(np.isfinite(v)) for v in sampled[name])
            out[name] = {
                "ci_low": ci[0],
                "ci_high": ci[1],
                "bootstrap_valid": valid,
                "bootstrap_samples": n_samples,
            }
    return out


def _ordinal_distance_matrix(categories: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    """Build Krippendorff ordinal distance matrix over categories."""
    n = len(categories)
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            low = i
            high = j
            segment = float(np.sum(frequencies[low: high + 1]))
            segment -= 0.5 * float(frequencies[low] + frequencies[high])
            delta = segment * segment
            dist[i, j] = delta
            dist[j, i] = delta
    return dist


def krippendorff_alpha_ordinal(ratings: np.ndarray) -> float:
    """Compute Krippendorff's alpha with ordinal disagreement.

    Args:
        ratings: 2D array [units, raters] with NaN for missing values.

    Returns:
        Alpha coefficient, or NaN if undefined.
    """
    if ratings.ndim != 2:
        raise ValueError("ratings must be 2D [units, raters]")

    flat = ratings[np.isfinite(ratings)]
    if flat.size < 2:
        return float("nan")

    categories, counts = np.unique(flat, return_counts=True)
    dist = _ordinal_distance_matrix(categories, counts.astype(float))

    cat_to_idx = {float(cat): idx for idx, cat in enumerate(categories)}

    do_num = 0.0
    do_den = 0.0
    for row in ratings:
        vals = row[np.isfinite(row)]
        m = len(vals)
        if m < 2:
            continue
        do_den += m * (m - 1) / 2.0
        idxs = [cat_to_idx[float(v)] for v in vals]
        for i in range(m):
            for j in range(i + 1, m):
                do_num += dist[idxs[i], idxs[j]]

    if do_den <= 0:
        return float("nan")
    do = do_num / do_den

    total = float(np.sum(counts))
    if total < 2:
        return float("nan")
    de_den = total * (total - 1.0) / 2.0
    de_num = 0.0
    for i in range(len(categories)):
        for j in range(i + 1, len(categories)):
            de_num += float(counts[i]) * float(counts[j]) * dist[i, j]

    if de_den <= 0:
        return float("nan")
    de = de_num / de_den

    if de <= 0:
        return 1.0 if do <= 0 else float("nan")
    return float(1.0 - (do / de))


def bootstrap_krippendorff_alpha_ordinal(
    ratings: np.ndarray,
    *,
    n_samples: int,
    random_seed: int,
) -> dict[str, float | int | None]:
    """Bootstrap CI for ordinal Krippendorff alpha by resampling units."""
    if ratings.ndim != 2:
        raise ValueError("ratings must be 2D [units, raters]")
    n_units = ratings.shape[0]
    point = krippendorff_alpha_ordinal(ratings)
    if n_units == 0:
        return {
            "alpha": point,
            "ci_low": None,
            "ci_high": None,
            "bootstrap_valid": 0,
            "bootstrap_samples": n_samples,
        }

    rng = np.random.default_rng(random_seed)
    samples: list[float] = []
    for _ in range(n_samples):
        idx = rng.integers(0, n_units, size=n_units)
        value = krippendorff_alpha_ordinal(ratings[idx, :])
        samples.append(value)

    ci = _bootstrap_ci(samples)
    if ci is None:
        return {
            "alpha": point,
            "ci_low": None,
            "ci_high": None,
            "bootstrap_valid": 0,
            "bootstrap_samples": n_samples,
        }

    return {
        "alpha": point,
        "ci_low": ci[0],
        "ci_high": ci[1],
        "bootstrap_valid": sum(int(np.isfinite(v)) for v in samples),
        "bootstrap_samples": n_samples,
    }


def run_drift_stats(ratings: np.ndarray) -> list[dict[str, Any]]:
    """Compute per-run drift diagnostics for a [units, runs] rating matrix."""
    if ratings.ndim != 2:
        raise ValueError("ratings must be 2D [units, runs]")

    rows: list[dict[str, Any]] = []
    for run_id in range(ratings.shape[1]):
        col = ratings[:, run_id]
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            rows.append(
                {
                    "run_id": run_id,
                    "n": 0,
                    "mean": float("nan"),
                    "stdev": float("nan"),
                    "min": float("nan"),
                    "max": float("nan"),
                }
            )
            continue
        rows.append(
            {
                "run_id": run_id,
                "n": int(finite.size),
                "mean": float(np.mean(finite)),
                "stdev": float(np.std(finite)),
                "min": float(np.min(finite)),
                "max": float(np.max(finite)),
            }
        )
    return rows
