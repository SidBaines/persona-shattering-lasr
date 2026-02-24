"""Tests for browser local chat in-memory session state."""

from __future__ import annotations

from scripts.visualisations.local_chat_web.state import SessionStore
from scripts.visualisations.local_chat_web.types import ChatAdapterConfig, GenerationSettings


def test_create_select_and_persist_chats() -> None:
    store = SessionStore()
    chat1 = store.create_chat(
        initial_adapters=[ChatAdapterConfig(key="o_avoiding", path="hf://org/a", scale=1.0)]
    )

    store.append_turn(
        chat1.chat_id,
        user_text="hello",
        assistant_text="hi",
        generation_settings=GenerationSettings(max_new_tokens=64, temperature=0.7, top_p=0.9),
    )

    chat2 = store.create_chat(initial_adapters=[])
    assert chat2.chat_id != chat1.chat_id

    sessions = store.list_sessions()
    assert [session.chat_id for session in sessions] == [chat1.chat_id, chat2.chat_id]

    store.select_chat(chat1.chat_id)
    selected = store.get_active_session()
    assert selected.chat_id == chat1.chat_id
    assert len(selected.turns) == 1
    assert selected.turns[0].assistant_text == "hi"


def test_turn_snapshots_preserve_per_turn_adapter_config() -> None:
    store = SessionStore()
    chat = store.create_chat(
        initial_adapters=[ChatAdapterConfig(key="adapter_a", path="hf://org/a", scale=1.0)]
    )

    store.append_turn(
        chat.chat_id,
        user_text="q1",
        assistant_text="a1",
        generation_settings=GenerationSettings(max_new_tokens=32, temperature=0.5, top_p=0.8),
    )

    store.set_adapter_scale(chat.chat_id, "adapter_a", 2.5)
    store.append_turn(
        chat.chat_id,
        user_text="q2",
        assistant_text="a2",
        generation_settings=GenerationSettings(max_new_tokens=64, temperature=0.9, top_p=0.95),
    )

    first_turn = store.get_session(chat.chat_id).turns[0]
    second_turn = store.get_session(chat.chat_id).turns[1]

    assert first_turn.adapter_config_snapshot[0].scale == 1.0
    assert second_turn.adapter_config_snapshot[0].scale == 2.5
