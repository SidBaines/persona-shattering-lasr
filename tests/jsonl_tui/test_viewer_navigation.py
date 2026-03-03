"""Tests for JSONL TUI navigation behavior."""

from __future__ import annotations

import curses

from scripts.jsonl_tui.viewer import JsonlViewer
from scripts.utils.io import write_jsonl


def _make_records() -> list[dict[str, object]]:
    return [
        {
            "sample_id": "sample-1",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ],
        },
        {
            "sample_id": "sample-2",
            "messages": [
                {"role": "user", "content": "second"},
                {"role": "assistant", "content": "record"},
            ],
        },
    ]


def test_conversation_mode_uses_arrow_keys_for_scrolling(tmp_path) -> None:
    path = write_jsonl(_make_records(), tmp_path / "records.jsonl")
    viewer = JsonlViewer(path, conversation_field="messages")
    viewer.line_offset = 2

    should_exit = viewer._handle_key(
        curses.KEY_DOWN,
        max_offset=10,
        body_height=5,
        current_group=viewer.grouped_records[viewer.question_index],
    )

    assert should_exit is False
    assert viewer.question_index == 0
    assert viewer.line_offset == 3


def test_conversation_mode_uses_n_and_p_for_record_navigation(tmp_path) -> None:
    path = write_jsonl(_make_records(), tmp_path / "records.jsonl")
    viewer = JsonlViewer(path, conversation_field="messages")

    viewer._handle_key(
        ord("n"),
        max_offset=10,
        body_height=5,
        current_group=viewer.grouped_records[viewer.question_index],
    )
    assert viewer.question_index == 1

    viewer._handle_key(
        ord("p"),
        max_offset=10,
        body_height=5,
        current_group=viewer.grouped_records[viewer.question_index],
    )
    assert viewer.question_index == 0


def test_default_mode_keeps_arrow_keys_for_record_navigation(tmp_path) -> None:
    path = write_jsonl(
        [
            {"question": "q1", "response": "r1"},
            {"question": "q2", "response": "r2"},
        ],
        tmp_path / "records.jsonl",
    )
    viewer = JsonlViewer(path)

    viewer._handle_key(
        curses.KEY_DOWN,
        max_offset=10,
        body_height=5,
        current_group=viewer.grouped_records[viewer.question_index],
    )

    assert viewer.question_index == 1
    assert viewer.line_offset == 0
