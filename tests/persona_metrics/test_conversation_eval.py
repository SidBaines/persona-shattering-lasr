"""Tests for MessageSelector negative turn_index_range support."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.persona_metrics.conversation_eval import (
    MessageSelector,
    _matches_selector,
    _resolve_turn_range,
    run_conversation_metrics,
    ConversationMetricsConfig,
)


# ── _resolve_turn_range ────────────────────────────────────────────────────────


def test_resolve_positive_indices_unchanged():
    assert _resolve_turn_range((0, 3), max_turn_index=4) == (0, 3)


def test_resolve_negative_lo():
    # max=4 → -1 → 4, -2 → 3
    assert _resolve_turn_range((-1, 4), max_turn_index=4) == (4, 4)
    assert _resolve_turn_range((-2, 4), max_turn_index=4) == (3, 4)


def test_resolve_negative_hi():
    assert _resolve_turn_range((0, -1), max_turn_index=4) == (0, 4)


def test_resolve_both_negative():
    assert _resolve_turn_range((-2, -1), max_turn_index=4) == (3, 4)


def test_resolve_last_turn_only():
    assert _resolve_turn_range((-1, -1), max_turn_index=9) == (9, 9)


def test_resolve_single_turn_conversation():
    assert _resolve_turn_range((-1, -1), max_turn_index=0) == (0, 0)


# ── _matches_selector ──────────────────────────────────────────────────────────


def _msg(role: str, turn_index: int | None, source_stage: str = "rollout_assistant") -> Any:
    meta: dict[str, Any] = {"source_stage": source_stage}
    if turn_index is not None:
        meta["turn_index"] = turn_index
    return SimpleNamespace(
        role=role,
        message_metadata=meta,
        message_id=f"{role}_{turn_index}",
    )


def test_no_filter_matches_all():
    sel = MessageSelector()
    assert _matches_selector(_msg("assistant", 0), sel, is_seed=False, max_turn_index=2)
    assert _matches_selector(_msg("user", 1), sel, is_seed=False, max_turn_index=2)


def test_exclude_seed():
    sel = MessageSelector(exclude_seed=True)
    assert not _matches_selector(_msg("user", 0), sel, is_seed=True, max_turn_index=2)
    assert _matches_selector(_msg("user", 0), sel, is_seed=False, max_turn_index=2)


def test_role_filter():
    sel = MessageSelector(roles=["assistant"])
    assert _matches_selector(_msg("assistant", 0), sel, is_seed=False, max_turn_index=2)
    assert not _matches_selector(_msg("user", 0), sel, is_seed=False, max_turn_index=2)


def test_absolute_turn_range():
    sel = MessageSelector(turn_index_range=(1, 2))
    assert not _matches_selector(_msg("assistant", 0), sel, is_seed=False, max_turn_index=3)
    assert _matches_selector(_msg("assistant", 1), sel, is_seed=False, max_turn_index=3)
    assert _matches_selector(_msg("assistant", 2), sel, is_seed=False, max_turn_index=3)
    assert not _matches_selector(_msg("assistant", 3), sel, is_seed=False, max_turn_index=3)


def test_negative_range_last_turn():
    # -1,-1 should only match the last turn
    sel = MessageSelector(turn_index_range=(-1, -1))
    assert not _matches_selector(_msg("assistant", 0), sel, is_seed=False, max_turn_index=2)
    assert not _matches_selector(_msg("assistant", 1), sel, is_seed=False, max_turn_index=2)
    assert _matches_selector(_msg("assistant", 2), sel, is_seed=False, max_turn_index=2)


def test_negative_range_last_two_turns():
    sel = MessageSelector(turn_index_range=(-2, -1))
    assert not _matches_selector(_msg("assistant", 0), sel, is_seed=False, max_turn_index=3)
    assert _matches_selector(_msg("assistant", 2), sel, is_seed=False, max_turn_index=3)
    assert _matches_selector(_msg("assistant", 3), sel, is_seed=False, max_turn_index=3)


def test_last_assistant_response_option_d():
    """Option (d): very last assistant response."""
    sel = MessageSelector(roles=["assistant"], turn_index_range=(-1, -1))
    assert not _matches_selector(_msg("user", 2), sel, is_seed=False, max_turn_index=2)
    assert not _matches_selector(_msg("assistant", 1), sel, is_seed=False, max_turn_index=2)
    assert _matches_selector(_msg("assistant", 2), sel, is_seed=False, max_turn_index=2)


def test_missing_turn_index_excluded_when_range_set():
    sel = MessageSelector(turn_index_range=(-1, -1))
    msg = SimpleNamespace(
        role="assistant",
        message_metadata={"source_stage": "rollout_assistant"},
        message_id="no_turn",
    )
    assert not _matches_selector(msg, sel, is_seed=False, max_turn_index=2)


# ── Integration: run_conversation_metrics respects negative selector ───────────


def _make_sample(turns: int) -> Any:
    """Build a minimal SampleRecord-like object with `turns` assistant turns."""
    messages = []
    for i in range(turns):
        messages.append(SimpleNamespace(
            role="user",
            content=f"user turn {i}",
            message_id=f"u{i}",
            message_metadata={"turn_index": i, "source_stage": "rollout_user_simulator"},
        ))
        messages.append(SimpleNamespace(
            role="assistant",
            content=f"assistant turn {i}",
            message_id=f"a{i}",
            message_metadata={"turn_index": i, "source_stage": "rollout_assistant"},
        ))
    return SimpleNamespace(sample_id="s0", messages=messages, input_group_id=None, response_index=0)


def test_integration_negative_selector_last_assistant_only(monkeypatch, tmp_path):
    """With roles=['assistant'] and turn_index_range=(-1,-1), only the final assistant
    message should be evaluated."""
    sample = _make_sample(turns=3)  # turns 0,1,2 → max_turn_index=2

    monkeypatch.setattr(
        "scripts.persona_metrics.conversation_eval.materialize_canonical_samples",
        lambda _: None,
    )
    monkeypatch.setattr(
        "scripts.persona_metrics.conversation_eval.load_samples",
        lambda _: [sample],
    )

    evaluated_contents: list[str] = []

    class _CapturingMetric:
        name = "capture"

        async def evaluate_batch_async(self, responses, questions, *, contexts=None):
            evaluated_contents.extend(responses)
            return [{"capture.score": 1}] * len(responses)

    monkeypatch.setattr(
        "scripts.persona_metrics.conversation_eval.get_persona_metric",
        lambda name, **_: _CapturingMetric(),
    )

    config = ConversationMetricsConfig(
        evaluations=["capture"],
        run_dir=tmp_path,
        message_selector=MessageSelector(
            roles=["assistant"],
            turn_index_range=(-1, -1),
        ),
    )
    result = run_conversation_metrics(config)

    assert result.num_messages_evaluated == 1
    assert evaluated_contents == ["assistant turn 2"]


def test_integration_whole_trajectory_assistant_only(monkeypatch, tmp_path):
    """roles=['assistant'] with no turn_index_range evaluates all assistant messages."""
    sample = _make_sample(turns=3)

    monkeypatch.setattr(
        "scripts.persona_metrics.conversation_eval.materialize_canonical_samples",
        lambda _: None,
    )
    monkeypatch.setattr(
        "scripts.persona_metrics.conversation_eval.load_samples",
        lambda _: [sample],
    )

    class _NoopMetric:
        name = "noop"

        async def evaluate_batch_async(self, responses, questions, *, contexts=None):
            return [{"noop.score": 0}] * len(responses)

    monkeypatch.setattr(
        "scripts.persona_metrics.conversation_eval.get_persona_metric",
        lambda name, **_: _NoopMetric(),
    )

    config = ConversationMetricsConfig(
        evaluations=["noop"],
        run_dir=tmp_path,
        message_selector=MessageSelector(roles=["assistant"]),
    )
    result = run_conversation_metrics(config)

    assert result.num_messages_evaluated == 3  # one per turn
