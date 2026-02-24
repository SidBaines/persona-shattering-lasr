"""Tests for prompt helpers used by browser local chat."""

from __future__ import annotations

from scripts.visualisations.local_chat_web.prompting import window_chat_turns
from scripts.visualisations.local_chat_web.types import ChatAdapterConfig, ChatTurn, GenerationSettings


def _turn(idx: int) -> ChatTurn:
    return ChatTurn(
        user_text=f"u{idx}",
        assistant_text=f"a{idx}",
        timestamp=f"t{idx}",
        adapter_config_snapshot=[ChatAdapterConfig(key="x", path="hf://org/a", scale=1.0)],
        generation_settings_snapshot=GenerationSettings(max_new_tokens=16, temperature=0.7, top_p=0.9),
    )


def test_window_chat_turns_respects_history_window() -> None:
    turns = [_turn(1), _turn(2), _turn(3)]
    windowed = window_chat_turns(turns, history_window=2)
    assert [turn.user_text for turn in windowed] == ["u2", "u3"]


def test_window_chat_turns_zero_or_negative_returns_full_history() -> None:
    turns = [_turn(1), _turn(2)]
    assert window_chat_turns(turns, history_window=0) == turns
    assert window_chat_turns(turns, history_window=-1) == turns
