"""Tests for canonical training candidate selection semantics."""

from __future__ import annotations

import pytest

from scripts.datasets import (
    ingest_source_dataset,
    select_training_candidates,
    write_edit_overlay,
    write_inference_result,
)


def test_select_training_candidates_fails_fast_on_incomplete_rows(tmp_path):
    run_dir = tmp_path / "run_training_select"
    samples = ingest_source_dataset(
        [{"question": "Q1"}, {"question": "Q2"}],
        source_info={"source": "test"},
        system_prompt="system",
        run_dir=run_dir,
        overwrite=True,
    )

    first, second = samples
    write_inference_result(
        run_dir,
        first.sample_id,
        {
            "status": "success",
            "model": "m",
            "provider": "p",
            "assistant_message_id": "msg_assistant_1",
            "assistant_completion": "A1",
            "assistant_full": "A1",
            "attempt_no": 1,
        },
        materialize=False,
    )
    write_edit_overlay(
        run_dir,
        sample_id=first.sample_id,
        variant_name="nano",
        overlay_payload={
            "overlay_id": "ov1",
            "target_message_id": "msg_assistant_1",
            "target_role": "assistant",
            "original_content_hash": "h1",
            "edited_content": "A1 edited",
            "status": "success",
            "attempt_no": 1,
        },
        materialize=False,
    )
    write_inference_result(
        run_dir,
        second.sample_id,
        {
            "status": "failed",
            "attempt_no": 1,
        },
        materialize=False,
    )

    with pytest.raises(ValueError, match="--skip-failed-rows"):
        _ = select_training_candidates(run_dir, "nano")


def test_select_training_candidates_can_skip_incomplete_rows(tmp_path):
    run_dir = tmp_path / "run_training_select_skip"
    samples = ingest_source_dataset(
        [{"question": "Q1"}, {"question": "Q2"}],
        source_info={"source": "test"},
        system_prompt="system",
        run_dir=run_dir,
        overwrite=True,
    )

    first, second = samples
    write_inference_result(
        run_dir,
        first.sample_id,
        {
            "status": "success",
            "model": "m",
            "provider": "p",
            "assistant_message_id": "msg_assistant_1",
            "assistant_completion": "A1",
            "assistant_full": "A1",
            "attempt_no": 1,
        },
        materialize=False,
    )
    write_edit_overlay(
        run_dir,
        sample_id=first.sample_id,
        variant_name="nano",
        overlay_payload={
            "overlay_id": "ov1",
            "target_message_id": "msg_assistant_1",
            "target_role": "assistant",
            "original_content_hash": "h1",
            "edited_content": "A1 edited",
            "status": "success",
            "attempt_no": 1,
        },
        materialize=False,
    )
    write_inference_result(
        run_dir,
        second.sample_id,
        {
            "status": "failed",
            "attempt_no": 1,
        },
        materialize=False,
    )

    dataset = select_training_candidates(run_dir, "nano", skip_failed_rows=True)
    assert len(dataset) == 1
    assert dataset[0]["sample_id"] == first.sample_id
