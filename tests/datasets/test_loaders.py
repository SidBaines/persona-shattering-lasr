"""Tests for dataset loading and inference formatting utilities."""

from __future__ import annotations

from datasets import Dataset

from scripts.common.config import DatasetConfig
from scripts.datasets import (
    format_for_inference,
    ingest_source_dataset,
    load_dataset_from_config,
    write_edit_overlay,
    write_inference_result,
)


def test_load_dataset_from_config_local_jsonl(tmp_path):
    local_path = tmp_path / "local.jsonl"
    local_path.write_text(
        "\n".join(
            [
                '{"question":"Q1","response":"R1"}',
                '{"question":"Q2","response":"R2"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = load_dataset_from_config(
        DatasetConfig(source="local", path=str(local_path), max_samples=1)
    )

    assert len(dataset) == 1
    assert dataset[0]["question"] == "Q1"
    assert dataset[0]["response"] == "R1"


def test_load_dataset_from_config_canonical_run_dir(tmp_path):
    run_dir = tmp_path / "run_canonical"
    samples = ingest_source_dataset(
        [{"question": "Q1"}, {"question": "Q2"}],
        source_info={"source": "test"},
        system_prompt="system",
        run_dir=run_dir,
        overwrite=True,
    )

    for idx, sample in enumerate(samples, start=1):
        write_inference_result(
            run_dir,
            sample.sample_id,
            {
                "status": "success",
                "model": "m",
                "provider": "p",
                "assistant_message_id": f"msg_assistant_{idx}",
                "assistant_completion": f"R{idx}",
                "assistant_full": f"R{idx}",
                "attempt_no": 1,
            },
            materialize=False,
        )

    dataset = load_dataset_from_config(
        DatasetConfig(source="canonical", path=str(run_dir))
    )

    assert len(dataset) == 2
    rows = dataset.to_list()
    questions = {row["question"] for row in rows}
    responses = {row["response"] for row in rows}
    assert questions == {"Q1", "Q2"}
    assert responses == {"R1", "R2"}


def test_load_dataset_from_config_canonical_editing_variant(tmp_path):
    run_dir = tmp_path / "run_variant"
    sample = ingest_source_dataset(
        [{"question": "Q1"}],
        source_info={"source": "test"},
        system_prompt="system",
        run_dir=run_dir,
        overwrite=True,
    )[0]
    write_inference_result(
        run_dir,
        sample.sample_id,
        {
            "status": "success",
            "model": "m",
            "provider": "p",
            "assistant_message_id": "msg_assistant_1",
            "assistant_completion": "R1",
            "assistant_full": "R1",
            "attempt_no": 1,
        },
        materialize=False,
    )
    write_edit_overlay(
        run_dir,
        sample_id=sample.sample_id,
        variant_name="nano",
        overlay_payload={
            "overlay_id": "ov_1",
            "target_message_id": "msg_assistant_1",
            "target_role": "assistant",
            "original_content_hash": "h1",
            "edited_content": "R1 edited",
            "status": "success",
            "attempt_no": 1,
        },
        materialize=False,
    )

    dataset = load_dataset_from_config(
        DatasetConfig(
            source="canonical",
            path=str(run_dir),
            name="editing:nano",
        )
    )

    assert len(dataset) == 1
    assert dataset[0]["response"] == "R1 edited"


def test_format_for_inference_detects_fallback_column():
    raw = Dataset.from_list([{"instruction": "What is 2+2?", "meta": 1}])
    formatted = format_for_inference(raw)

    assert formatted.column_names == ["question"]
    assert formatted[0]["question"] == "What is 2+2?"


def test_format_for_inference_merges_input_and_drops_extra_columns():
    raw = Dataset.from_list(
        [{"question": "Summarize this", "input": "A short paragraph.", "id": "x"}]
    )
    formatted = format_for_inference(raw)

    assert formatted.column_names == ["question"]
    assert formatted[0]["question"] == "Summarize this\n\nA short paragraph."


def test_load_dataset_from_config_rejects_unsupported_source():
    config = DatasetConfig(source="local")
    config.source = "invalid"
    try:
        _ = load_dataset_from_config(config)
    except ValueError as exc:
        assert "Unsupported dataset source" in str(exc)
        return
    raise AssertionError("Expected ValueError for unsupported dataset source")


def test_load_dataset_from_config_rejects_missing_required_fields():
    try:
        _ = load_dataset_from_config(DatasetConfig(source="huggingface", name=None))
    except ValueError as exc:
        assert "requires dataset name" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing HuggingFace dataset name")

    try:
        _ = load_dataset_from_config(DatasetConfig(source="local", path=None))
    except ValueError as exc:
        assert "requires dataset path" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing local dataset path")

    try:
        _ = load_dataset_from_config(DatasetConfig(source="canonical", path=None))
    except ValueError as exc:
        assert "requires run directory path" in str(exc)
        return
    raise AssertionError("Expected ValueError for missing canonical run directory path")


def test_format_for_inference_rejects_missing_question_column():
    raw = Dataset.from_list([{"title": "Q1"}])
    try:
        _ = format_for_inference(raw)
    except ValueError as exc:
        assert "Could not find question column" in str(exc)
        return
    raise AssertionError("Expected ValueError for missing question column")
