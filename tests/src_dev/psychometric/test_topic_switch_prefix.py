"""Regression test for the trait_mcq topic-switch prefix.

The prefix is a user-turn rendering feature added to signal an explicit
topic switch between a rollout conversation and a trait_mcq probe. The
rollout's chat-template turn boundary already separates content, but on
long coherent rollouts (e.g. the B preset's 15-turn scenarios) a plain
question can still read as a continuation. Prepending ``"New question: "``
(see ``TRAIT_MCQ_TOPIC_SWITCH_PREFIX`` in
``src_dev/psychometric/item_prompts.py``) is belt-and-suspenders.

Tests here guard against two regressions:

1. Default behaviour is unchanged — ``build_trait_mcq_prompt(item)`` and
   ``build_item_prompt(item)`` without the flag produce the same output
   they did before the feature landed. This protects run-id stability
   for existing HF-cached trait_mcq runs (``trait_ocean_v1``,
   ``trait_ocean_v1_nolead``) that were produced under the old
   rendering — any silent change would desync the cache keys.

2. When the flag is passed, the rendered text starts with the prefix
   and is otherwise identical to the default rendering. The flag is a
   no-op for non-trait_mcq item types.
"""

from __future__ import annotations

from src_dev.psychometric.item_prompts import (
    TRAIT_MCQ_TOPIC_SWITCH_PREFIX,
    build_item_prompt,
    build_trait_mcq_prompt,
)


def _trait_mcq_item() -> dict:
    return {
        "id": "openness_test_item",
        "type": "trait_mcq",
        "question": "How should I spend my afternoon?",
        "options": [
            {"label": "A", "text": "Read a familiar book."},
            {"label": "B", "text": "Try a craft I've never done before."},
            {"label": "C", "text": "Watch a popular movie everyone recommends."},
            {"label": "D", "text": "Explore a neighbourhood I've never been to."},
        ],
        "answer_mapping": {"A": 0, "B": 1, "C": 0, "D": 1},
        "primary_dimension": "openness",
    }


def test_default_rendering_has_no_topic_switch_prefix():
    """Default call (no flag) must not include the prefix.

    Run-id stability for existing trait_mcq HF caches depends on the
    default rendering matching the pre-feature output.
    """
    item = _trait_mcq_item()
    rendered = build_trait_mcq_prompt(item)
    assert not rendered.startswith(TRAIT_MCQ_TOPIC_SWITCH_PREFIX), (
        f"Default build_trait_mcq_prompt rendering must NOT include the "
        f"topic-switch prefix. Got:\n{rendered!r}"
    )
    # Question text should appear as-is at the start.
    assert rendered.startswith(item["question"]), (
        f"Default rendering should start with the question text. Got:\n{rendered!r}"
    )


def test_flag_prepends_topic_switch_prefix():
    """With ``topic_switch_prefix=True``, output begins with the prefix.

    The rest of the output must match the default rendering exactly,
    confirming the flag is purely additive — the prefix is prepended,
    nothing else changes.
    """
    item = _trait_mcq_item()
    default = build_trait_mcq_prompt(item)
    with_prefix = build_trait_mcq_prompt(item, topic_switch_prefix=True)
    assert with_prefix.startswith(TRAIT_MCQ_TOPIC_SWITCH_PREFIX), (
        f"Expected rendering to start with "
        f"{TRAIT_MCQ_TOPIC_SWITCH_PREFIX!r}; got:\n{with_prefix!r}"
    )
    assert with_prefix == TRAIT_MCQ_TOPIC_SWITCH_PREFIX + default, (
        f"Flag should only prepend the prefix, not alter the rest.\n"
        f"  default     = {default!r}\n"
        f"  with_prefix = {with_prefix!r}"
    )


def test_dispatch_via_build_item_prompt():
    """``build_item_prompt`` must forward the flag to the trait_mcq builder."""
    item = _trait_mcq_item()
    dispatched = build_item_prompt(item, trait_mcq_topic_switch_prefix=True)
    direct = build_trait_mcq_prompt(item, topic_switch_prefix=True)
    assert dispatched == direct, (
        "build_item_prompt must forward trait_mcq_topic_switch_prefix to "
        "build_trait_mcq_prompt.\n"
        f"  dispatched = {dispatched!r}\n"
        f"  direct     = {direct!r}"
    )


def test_flag_is_noop_for_non_trait_mcq():
    """Non-trait_mcq item types must ignore the flag (no-op)."""
    likert_item = {
        "id": "lik_1",
        "type": "likert",
        "text": "I enjoy meeting new people.",
    }
    without = build_item_prompt(likert_item)
    with_flag = build_item_prompt(likert_item, trait_mcq_topic_switch_prefix=True)
    assert without == with_flag, (
        "topic_switch_prefix flag should be a no-op for non-trait_mcq "
        "items but the rendered output changed:\n"
        f"  without  = {without!r}\n"
        f"  with_flag = {with_flag!r}"
    )
