"""Tests for multi-turn canonical dataset helpers."""

from __future__ import annotations

from scripts.datasets import (
    export_dataset,
    ingest_source_dataset,
    load_samples,
    render_messages,
    write_edit_overlay,
    write_inference_result,
    write_message_append,
)


def test_render_messages_applies_overlay_to_target_message_only(tmp_path):
    run_dir = tmp_path / "run_render"
    sample = ingest_source_dataset(
        [{"question": "Q1"}],
        source_info={"source": "test"},
        system_prompt=None,
        run_dir=run_dir,
        overwrite=True,
    )[0]
    write_inference_result(
        run_dir,
        sample.sample_id,
        {
            "status": "success",
            "model": "base",
            "provider": "local",
            "assistant_message_id": "msg_assistant_1",
            "assistant_completion": "A1 base",
            "assistant_full": "A1 base",
            "assistant_message_metadata": {"turn_index": 0},
            "attempt_no": 1,
        },
        materialize=False,
    )
    write_message_append(
        run_dir,
        sample.sample_id,
        {
            "message_id": "msg_user_2",
            "role": "user",
            "content": "Q2",
            "message_metadata": {"turn_index": 1},
        },
        materialize=False,
    )
    write_message_append(
        run_dir,
        sample.sample_id,
        {
            "message_id": "msg_assistant_2",
            "role": "assistant",
            "content": "A2 base",
            "message_metadata": {"turn_index": 1},
        },
        materialize=False,
    )
    write_edit_overlay(
        run_dir,
        sample_id=sample.sample_id,
        variant_name="multiturn",
        overlay_payload={
            "overlay_id": "ov_2",
            "target_message_id": "msg_assistant_2",
            "target_role": "assistant",
            "original_content_hash": "h2",
            "edited_content": "A2 edited",
            "status": "success",
            "attempt_no": 1,
        },
        materialize=False,
    )

    materialized = load_samples(run_dir)[0]
    rendered = render_messages(materialized, "multiturn")

    assistant_text = [message.content for message in rendered if message.role == "assistant"]
    assert assistant_text == ["A1 base", "A2 edited"]


def test_export_conversation_training_uses_rendered_messages(tmp_path):
    run_dir = tmp_path / "run_export"
    sample = ingest_source_dataset(
        [{"question": "Q1"}],
        source_info={"source": "test"},
        system_prompt=None,
        run_dir=run_dir,
        overwrite=True,
    )[0]
    write_inference_result(
        run_dir,
        sample.sample_id,
        {
            "status": "success",
            "model": "base",
            "provider": "local",
            "assistant_message_id": "msg_assistant_1",
            "assistant_completion": "A1 base",
            "assistant_full": "A1 base",
            "attempt_no": 1,
        },
        materialize=False,
    )
    write_edit_overlay(
        run_dir,
        sample_id=sample.sample_id,
        variant_name="multiturn",
        overlay_payload={
            "overlay_id": "ov_1",
            "target_message_id": "msg_assistant_1",
            "target_role": "assistant",
            "original_content_hash": "h1",
            "edited_content": "A1 edited",
            "status": "success",
            "attempt_no": 1,
        },
        materialize=False,
    )

    export_path = export_dataset(
        run_dir,
        profile="conversation_training",
        variant_name="multiturn",
    )

    row = export_path.read_text(encoding="utf-8").strip()
    assert '"editing_variant": "multiturn"' in row
    assert '"content": "A1 edited"' in row
