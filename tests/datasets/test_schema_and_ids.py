"""Tests for canonical dataset schema and deterministic IDs."""

from __future__ import annotations

from pydantic import ValidationError

from scripts.datasets import ingest_source_dataset
from scripts.datasets.schema import CanonicalMessage


def test_canonical_message_rejects_invalid_role():
    try:
        CanonicalMessage(message_id="msg_1", role="invalid", content="x")
    except ValidationError:
        return
    raise AssertionError("Expected ValidationError for invalid message role")


def test_sample_ids_are_deterministic_across_row_order(tmp_path):
    rows_a = [
        {"question": "Q1", "input": "extra"},
        {"question": "Q2"},
    ]
    rows_b = [
        {"question": "Q2"},
        {"question": "Q1", "input": "extra"},
    ]

    run_dir_a = tmp_path / "run_a"
    run_dir_b = tmp_path / "run_b"
    samples_a = ingest_source_dataset(
        rows_a,
        source_info={"source": "test"},
        system_prompt="You are helpful.",
        run_dir=run_dir_a,
        overwrite=True,
    )
    samples_b = ingest_source_dataset(
        rows_b,
        source_info={"source": "test"},
        system_prompt="You are helpful.",
        run_dir=run_dir_b,
        overwrite=True,
    )

    by_question_a = {
        next(msg.content for msg in sample.messages if msg.role == "user"): sample.sample_id
        for sample in samples_a
    }
    by_question_b = {
        next(msg.content for msg in sample.messages if msg.role == "user"): sample.sample_id
        for sample in samples_b
    }

    assert by_question_a == by_question_b


def test_responses_per_input_expands_rows_with_shared_group_id(tmp_path):
    run_dir = tmp_path / "run_multi_response"
    samples = ingest_source_dataset(
        [{"question": "Q1"}],
        source_info={"source": "test"},
        system_prompt=None,
        run_dir=run_dir,
        overwrite=True,
        responses_per_input=3,
    )

    assert len(samples) == 3
    group_ids = {sample.input_group_id for sample in samples}
    assert len(group_ids) == 1
    sample_ids = {sample.sample_id for sample in samples}
    assert len(sample_ids) == 3
    response_indexes = sorted(sample.response_index for sample in samples)
    assert response_indexes == [0, 1, 2]
