"""Tests for construct-validity calculations."""

from __future__ import annotations

import numpy as np

from scripts.calibration.run import _aggregate_analysis_units
from scripts.calibration.statistics import validity_metrics


def test_validity_metrics_recovers_linear_relationship() -> None:
    rng = np.random.default_rng(19)
    judge_z = np.linspace(-2.0, 2.0, 200)
    gt_z = 0.5 + 1.2 * judge_z + rng.normal(0.0, 0.08, size=judge_z.shape[0])

    metrics = validity_metrics(judge_z, gt_z)

    assert metrics["pearson_r"] > 0.98
    assert metrics["spearman_rho"] > 0.98
    assert abs(metrics["slope"] - 1.2) < 0.05
    assert abs(metrics["intercept"] - 0.5) < 0.08
    assert metrics["r2"] > 0.95


def test_subject_analysis_aggregation_reduces_to_subject_count() -> None:
    text_rows = [
        {
            "unit_id": "u1",
            "subject_id": "s1",
            "mean_raw": -1.0,
            "std_raw": 0.1,
            "label_raw": -0.8,
        },
        {
            "unit_id": "u2",
            "subject_id": "s1",
            "mean_raw": -0.5,
            "std_raw": 0.2,
            "label_raw": -0.8,
        },
        {
            "unit_id": "u3",
            "subject_id": "s2",
            "mean_raw": 1.0,
            "std_raw": 0.1,
            "label_raw": 1.2,
        },
        {
            "unit_id": "u4",
            "subject_id": "s2",
            "mean_raw": 0.8,
            "std_raw": 0.1,
            "label_raw": 1.2,
        },
    ]

    aggregated = _aggregate_analysis_units(text_rows, analysis_unit="subject")

    assert len(aggregated) == 2
    by_subject = {row["analysis_unit_id"]: row for row in aggregated}
    assert by_subject["s1"]["n_text_units"] == 2
    assert by_subject["s2"]["n_text_units"] == 2
    assert np.isclose(by_subject["s1"]["mean_raw"], -0.75)
    assert np.isclose(by_subject["s2"]["mean_raw"], 0.9)
