"""Tests for canonical resume and materialization behavior."""

from __future__ import annotations

from scripts.datasets import (
    ingest_source_dataset,
    load_samples,
    materialize_canonical_samples,
    resume_state,
    write_edit_overlay,
    write_inference_result,
)


def test_resume_state_tracks_inference_and_editing(tmp_path):
    run_dir = tmp_path / "run_resume"
    samples = ingest_source_dataset(
        [{"question": "Q1"}, {"question": "Q2"}],
        source_info={"source": "test"},
        system_prompt="system",
        run_dir=run_dir,
        overwrite=True,
    )

    state_before = resume_state(run_dir, "inference")
    assert len(state_before["pending"]) == 2
    assert not state_before["complete"]

    first = samples[0]
    write_inference_result(
        run_dir,
        first.sample_id,
        {
            "status": "success",
            "model": "m",
            "provider": "p",
            "assistant_message_id": "msg_assistant",
            "assistant_completion": "A1",
            "assistant_full": "A1",
            "attempt_no": 1,
        },
        materialize=False,
    )
    materialize_canonical_samples(run_dir)

    state_after = resume_state(run_dir, "inference")
    assert first.sample_id in state_after["complete"]
    assert len(state_after["pending"]) == 1

    write_edit_overlay(
        run_dir,
        sample_id=first.sample_id,
        variant_name="nano",
        overlay_payload={
            "overlay_id": "ov1",
            "target_message_id": "msg_assistant",
            "target_role": "assistant",
            "original_content_hash": "h1",
            "edited_content": "A1 edited",
            "status": "success",
            "attempt_no": 1,
        },
        materialize=False,
    )
    materialize_canonical_samples(run_dir)

    editing_state = resume_state(run_dir, "editing", "nano")
    assert first.sample_id in editing_state["complete"]

    canonical = load_samples(run_dir)
    first_row = next(sample for sample in canonical if sample.sample_id == first.sample_id)
    variant = next(v for v in first_row.edit_variants if v.variant_name == "nano")
    assert variant.status == "success"
    assert variant.overlays[0].edited_content == "A1 edited"

