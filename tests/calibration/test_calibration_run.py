"""End-to-end tests for calibration run orchestration."""

from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset

from scripts.calibration.config import (
    CalibrationConfig,
    CalibrationDatasetConfig,
    CalibrationJudgeConfig,
    ReliabilityConfig,
    ReliabilityRaterConfig,
    ValidityConfig,
)
from scripts.calibration.run import run_calibration
from scripts.calibration.traits import get_trait_preset
from scripts.common.config import DatasetConfig


class _MockMetric:
    def __init__(self, score_vectors: list[list[float]], call_index_ref: dict[str, int]) -> None:
        self._score_vectors = score_vectors
        self._call_index_ref = call_index_ref

    async def evaluate_batch_async(self, responses, questions, contexts=None):
        run_id = self._call_index_ref["value"]
        self._call_index_ref["value"] += 1
        scores = self._score_vectors[run_id]

        out = []
        for idx, _ in enumerate(responses):
            out.append(
                {
                    "neuroticism.score": scores[idx],
                    "neuroticism.reasoning": f"mock reasoning run={run_id} row={idx}",
                }
            )
        return out


def test_run_calibration_writes_expected_artifacts(tmp_path: Path, monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "sample_id": "u1",
                "subject_id": "s1",
                "question": "Q1",
                "response": "R1",
                "neuroticism": -1.0,
            },
            {
                "sample_id": "u2",
                "subject_id": "s1",
                "question": "Q2",
                "response": "R2",
                "neuroticism": -0.5,
            },
            {
                "sample_id": "u3",
                "subject_id": "s2",
                "question": "Q3",
                "response": "R3",
                "neuroticism": 1.0,
            },
            {
                "sample_id": "u4",
                "subject_id": "s2",
                "question": "Q4",
                "response": "R4",
                "neuroticism": 1.5,
            },
        ]
    )

    score_vectors = [
        [-1.0, -0.5, 1.0, 1.5],
        [-1.2, -0.4, 1.1, 1.6],
        [-0.9, -0.3, 1.2, 1.4],
    ]
    call_index_ref = {"value": 0}

    def _fake_get_persona_metric(name: str, **kwargs):
        _ = (name, kwargs)
        return _MockMetric(score_vectors, call_index_ref)

    monkeypatch.setattr("scripts.calibration.run.get_persona_metric", _fake_get_persona_metric)

    config = CalibrationConfig(
        dataset=CalibrationDatasetConfig(
            dataset=DatasetConfig(source="local", path="unused.jsonl"),
            response_column="response",
            question_column="question",
            label_column="neuroticism",
            subject_id_column="subject_id",
            unit_id_column="sample_id",
        ),
        judge=CalibrationJudgeConfig(metric_name="neuroticism"),
        trait=get_trait_preset("neuroticism"),
        reliability=ReliabilityConfig(num_runs=3, bootstrap_samples=50, min_units=1),
        validity=ValidityConfig(analysis_unit="subject", bootstrap_samples=50),
        output_dir=tmp_path / "calibration_run",
    )

    result_dataset, result = run_calibration(config, dataset=dataset)

    assert result.analysis_unit == "subject"
    assert result.num_scored_units == 2
    assert call_index_ref["value"] == 3

    for key in ["scores_long", "scores_unit", "reliability", "validity", "summary", "report"]:
        assert key in result.artifacts
        assert result.artifacts[key].exists()

    assert len(result_dataset) == 2
    assert "judge_z" in result_dataset.column_names
    assert "gt_z" in result_dataset.column_names

    summary = json.loads((tmp_path / "calibration_run" / "summary.json").read_text(encoding="utf-8"))
    assert summary["analysis_unit"] == "subject"
    assert summary["reliability"]["num_runs"] == 3
    assert summary["reliability"]["protocol"] == "repeat_runs"
    assert "pearson_r" in summary["validity"]["point"]
    assert "pearson_r" in summary["validity"]["point_raw"]
    assert "pearson_r" in summary["validity"]["point_z"]


def test_run_calibration_keeps_missing_labels_out_of_validity_pairs(
    tmp_path: Path, monkeypatch
) -> None:
    dataset = Dataset.from_list(
        [
            {"sample_id": "u1", "response": "R1", "question": "Q1", "neuroticism": -1.0},
            {"sample_id": "u2", "response": "R2", "question": "Q2", "neuroticism": None},
            {"sample_id": "u3", "response": "R3", "question": "Q3", "neuroticism": 1.0},
        ]
    )
    score_vectors = [[-1.1, 0.2, 1.1], [-1.0, 0.4, 1.2]]
    call_index_ref = {"value": 0}

    def _fake_get_persona_metric(name: str, **kwargs):
        _ = (name, kwargs)
        return _MockMetric(score_vectors, call_index_ref)

    monkeypatch.setattr("scripts.calibration.run.get_persona_metric", _fake_get_persona_metric)

    config = CalibrationConfig(
        dataset=CalibrationDatasetConfig(
            dataset=DatasetConfig(source="local", path="unused.jsonl"),
            response_column="response",
            question_column="question",
            label_column="neuroticism",
            unit_id_column="sample_id",
        ),
        judge=CalibrationJudgeConfig(metric_name="neuroticism"),
        trait=get_trait_preset("neuroticism"),
        reliability=ReliabilityConfig(num_runs=2, bootstrap_samples=20, min_units=1),
        validity=ValidityConfig(analysis_unit="text", bootstrap_samples=20),
        output_dir=tmp_path / "calibration_missing_labels",
    )

    _, result = run_calibration(config, dataset=dataset)
    assert result.validity["num_units_total"] == 3
    assert result.validity["num_units_valid"] == 2
    assert result.validity["point"]["n"] == 2.0
    validity_text = (tmp_path / "calibration_missing_labels" / "validity.json").read_text(
        encoding="utf-8"
    )
    unit_text = (tmp_path / "calibration_missing_labels" / "scores_unit.jsonl").read_text(
        encoding="utf-8"
    )
    assert "NaN" not in validity_text
    assert "NaN" not in unit_text


def test_run_calibration_auto_analysis_unit_falls_back_to_text_on_partial_subject_ids(
    tmp_path: Path, monkeypatch
) -> None:
    dataset = Dataset.from_list(
        [
            {"sample_id": "u1", "subject_id": "s1", "response": "R1", "question": "Q1", "neuroticism": -1.0},
            {"sample_id": "u2", "subject_id": None, "response": "R2", "question": "Q2", "neuroticism": -0.2},
            {"sample_id": "u3", "subject_id": "s2", "response": "R3", "question": "Q3", "neuroticism": 1.0},
        ]
    )
    score_vectors = [[-1.0, -0.1, 1.2], [-1.1, -0.3, 1.1]]
    call_index_ref = {"value": 0}

    def _fake_get_persona_metric(name: str, **kwargs):
        _ = (name, kwargs)
        return _MockMetric(score_vectors, call_index_ref)

    monkeypatch.setattr("scripts.calibration.run.get_persona_metric", _fake_get_persona_metric)

    config = CalibrationConfig(
        dataset=CalibrationDatasetConfig(
            dataset=DatasetConfig(source="local", path="unused.jsonl"),
            response_column="response",
            question_column="question",
            label_column="neuroticism",
            subject_id_column="subject_id",
            unit_id_column="sample_id",
        ),
        judge=CalibrationJudgeConfig(metric_name="neuroticism"),
        trait=get_trait_preset("neuroticism"),
        reliability=ReliabilityConfig(num_runs=2, bootstrap_samples=20, min_units=1),
        validity=ValidityConfig(analysis_unit="auto", bootstrap_samples=20),
        output_dir=tmp_path / "calibration_partial_subjects",
    )

    result_dataset, result = run_calibration(config, dataset=dataset)
    assert result.analysis_unit == "text"
    assert len(result_dataset) == 3
    assert any("partial subject IDs" in warning for warning in result.warnings)


def test_run_calibration_uses_configured_raters_for_reliability(
    tmp_path: Path, monkeypatch
) -> None:
    dataset = Dataset.from_list(
        [
            {"sample_id": "u1", "response": "R1", "question": "Q1", "neuroticism": -1.0},
            {"sample_id": "u2", "response": "R2", "question": "Q2", "neuroticism": 1.0},
        ]
    )
    score_vectors = [[-1.0, 1.0], [-0.7, 0.8]]
    call_index_ref = {"value": 0}

    def _fake_get_persona_metric(name: str, **kwargs):
        _ = (name, kwargs)
        return _MockMetric(score_vectors, call_index_ref)

    monkeypatch.setattr("scripts.calibration.run.get_persona_metric", _fake_get_persona_metric)

    config = CalibrationConfig(
        dataset=CalibrationDatasetConfig(
            dataset=DatasetConfig(source="local", path="unused.jsonl"),
            response_column="response",
            question_column="question",
            label_column="neuroticism",
            unit_id_column="sample_id",
        ),
        judge=CalibrationJudgeConfig(metric_name="neuroticism"),
        trait=get_trait_preset("neuroticism"),
        reliability=ReliabilityConfig(
            num_runs=7,
            raters=[
                ReliabilityRaterConfig(name="judge_a"),
                ReliabilityRaterConfig(
                    name="judge_b",
                    metric_params={"include_reasoning": False},
                ),
            ],
            bootstrap_samples=20,
            min_units=1,
        ),
        validity=ValidityConfig(analysis_unit="text", bootstrap_samples=20),
        output_dir=tmp_path / "calibration_configured_raters",
    )

    _, result = run_calibration(config, dataset=dataset)

    assert call_index_ref["value"] == 2
    assert result.reliability["protocol"] == "configured_raters"
    assert result.reliability["num_runs"] == 2
    assert len(result.reliability["raters"]) == 2
    assert result.reliability["raters"][0]["rater_name"] == "judge_a"
    assert result.reliability["raters"][1]["rater_name"] == "judge_b"
