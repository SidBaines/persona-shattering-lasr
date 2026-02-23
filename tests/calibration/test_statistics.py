"""Tests for calibration statistics helpers."""

from __future__ import annotations

import numpy as np

from scripts.calibration.statistics import (
    bootstrap_krippendorff_alpha_ordinal,
    krippendorff_alpha_ordinal,
    zscore,
)


def test_krippendorff_alpha_ordinal_is_one_for_perfect_agreement() -> None:
    ratings = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [-1.0, -1.0, -1.0],
            [0.0, 0.0, 0.0],
        ]
    )
    alpha = krippendorff_alpha_ordinal(ratings)
    assert alpha == 1.0


def test_krippendorff_alpha_ordinal_near_zero_for_random_scores() -> None:
    rng = np.random.default_rng(7)
    ratings = rng.integers(-2, 3, size=(300, 4)).astype(float)
    alpha = krippendorff_alpha_ordinal(ratings)
    assert -0.15 <= alpha <= 0.15


def test_krippendorff_alpha_ordinal_handles_missing_values() -> None:
    ratings = np.array(
        [
            [1.0, 1.0, np.nan],
            [2.0, np.nan, 2.0],
            [0.0, 1.0, 0.0],
            [np.nan, np.nan, 3.0],
        ]
    )
    alpha = krippendorff_alpha_ordinal(ratings)
    assert np.isfinite(alpha)


def test_bootstrap_krippendorff_ci_has_bounds() -> None:
    rng = np.random.default_rng(11)
    ratings = rng.integers(-3, 4, size=(120, 5)).astype(float)
    summary = bootstrap_krippendorff_alpha_ordinal(
        ratings,
        n_samples=120,
        random_seed=5,
    )

    assert "alpha" in summary
    assert "ci_low" in summary
    assert "ci_high" in summary
    assert summary["bootstrap_valid"] <= summary["bootstrap_samples"]
    assert summary["ci_low"] is not None
    assert summary["ci_high"] is not None
    assert summary["ci_low"] <= summary["ci_high"]


def test_zscore_preserves_missing_values() -> None:
    values = np.array([1.0, np.nan, 1.0, np.inf, -np.inf], dtype=float)
    z, mean, std, fallback = zscore(values)

    assert fallback is True
    assert mean == 1.0
    assert std == 0.0
    assert z.shape == values.shape
    assert z[0] == 0.0
    assert z[2] == 0.0
    assert np.isnan(z[1])
    assert np.isnan(z[3])
    assert np.isnan(z[4])
