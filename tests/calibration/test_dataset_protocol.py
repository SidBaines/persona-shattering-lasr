"""Tests for calibration dataset protocol adapters."""

from __future__ import annotations

from datasets import Dataset

from scripts.calibration.config import CalibrationConfig, CalibrationDatasetConfig
from scripts.calibration.datasets import apply_calibration_dataset_protocol
from scripts.calibration.traits import get_trait_preset
from scripts.common.config import DatasetConfig


def test_profile_mapping_and_linear_normalization_to_trait_range() -> None:
    dataset = Dataset.from_list(
        [
            {"essay_text": "text one", "neuroticism_score": 1.0, "author_id": "a1"},
            {"essay_text": "text two", "neuroticism_score": 3.0, "author_id": "a2"},
            {"essay_text": "text three", "neuroticism_score": 5.0, "author_id": "a3"},
        ]
    )

    config = CalibrationConfig(
        dataset=CalibrationDatasetConfig(
            dataset=DatasetConfig(source="local", path="unused.jsonl"),
            response_column="response",
            question_column="question",
            label_column="label",
            subject_id_column="subject_id",
            dataset_profile="essays_neuroticism_v1",
            split=CalibrationDatasetConfig.CalibrationSplitConfig(eval_split="all"),
            normalization=CalibrationDatasetConfig.LabelNormalizationConfig(
                mode="linear_to_trait_range",
                source_min=1.0,
                source_max=5.0,
                clip_to_trait_range=True,
            ),
        ),
        trait=get_trait_preset("neuroticism"),
    )

    warnings: list[str] = []
    out, metadata = apply_calibration_dataset_protocol(dataset, config, warnings=warnings)
    rows = out.to_list()

    assert len(rows) == 3
    assert [row["response"] for row in rows] == ["text one", "text two", "text three"]
    assert [row["label"] for row in rows] == [-5.0, 0.0, 5.0]
    assert [row["subject_id"] for row in rows] == ["a1", "a2", "a3"]
    assert all("sample_id" in row for row in rows)
    assert metadata["profile"] == "essays_neuroticism_v1"
    assert metadata["normalization"]["mode"] == "linear_to_trait_range"
    assert warnings


def test_deterministic_split_filtering_is_reproducible() -> None:
    dataset = Dataset.from_list(
        [
            {"essay_text": f"text {idx}", "neuroticism_score": float(idx % 5), "author_id": f"a{idx}"}
            for idx in range(30)
        ]
    )
    split_cfg = CalibrationDatasetConfig.CalibrationSplitConfig(
        eval_split="test",
        split_seed=11,
        train_fraction=0.6,
        dev_fraction=0.2,
    )
    config = CalibrationConfig(
        dataset=CalibrationDatasetConfig(
            dataset=DatasetConfig(source="local", path="unused.jsonl"),
            response_column="response",
            label_column="label",
            question_column="question",
            dataset_profile="essays_neuroticism_v1",
            split=split_cfg,
        ),
        trait=get_trait_preset("neuroticism"),
    )

    warnings_a: list[str] = []
    warnings_b: list[str] = []
    out_a, meta_a = apply_calibration_dataset_protocol(dataset, config, warnings=warnings_a)
    out_b, meta_b = apply_calibration_dataset_protocol(dataset, config, warnings=warnings_b)

    ids_a = [row["sample_id"] for row in out_a.to_list()]
    ids_b = [row["sample_id"] for row in out_b.to_list()]
    assert ids_a == ids_b
    assert meta_a["split_counts_all_rows"] == meta_b["split_counts_all_rows"]
    assert sum(meta_a["split_counts_all_rows"].values()) == 30
    assert meta_a["output_rows"] == len(ids_a)


def test_explicit_split_column_alias_maps_validation_to_dev() -> None:
    dataset = Dataset.from_list(
        [
            {"essay_text": "a", "neuroticism_score": 0.1, "fold": "train"},
            {"essay_text": "b", "neuroticism_score": 0.2, "fold": "validation"},
            {"essay_text": "c", "neuroticism_score": 0.3, "fold": "test"},
        ]
    )

    config = CalibrationConfig(
        dataset=CalibrationDatasetConfig(
            dataset=DatasetConfig(source="local", path="unused.jsonl"),
            response_column="response",
            label_column="label",
            question_column="question",
            dataset_profile="essays_neuroticism_v1",
            split=CalibrationDatasetConfig.CalibrationSplitConfig(eval_split="dev"),
        ),
        trait=get_trait_preset("neuroticism"),
    )

    warnings: list[str] = []
    out, metadata = apply_calibration_dataset_protocol(dataset, config, warnings=warnings)
    rows = out.to_list()

    assert len(rows) == 1
    assert rows[0]["response"] == "b"
    assert rows[0]["_calibration_split"] == "dev"
    assert metadata["split_column_used"] == "fold"
