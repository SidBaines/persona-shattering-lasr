"""Tests for conversation-reset prompt builders.

These tests cover the three reset modes used by the psychometric-FA
questionnaire stage: ``none`` (baseline), ``soft`` (mid-context system
reset), and ``token_boundary`` (raw-token ``<|end_of_text|>`` / ``<|begin_of_text|>``
split).

The messages-level builder is tested with a pure-Python fixture (no
tokenizer). The ``token_boundary`` builder uses a real HuggingFace tokenizer
so that the chat-template / special-token behaviour is exercised end-to-end;
each tokenizer family we care about is surfaced as a parameter so adding a
new model (e.g. qwen) means just adding an entry and re-running the tests.
"""

from __future__ import annotations

from typing import Any

import pytest

from src_dev.inference.conversation_reset import (
    DEFAULT_SOFT_RESET_SYSTEM_PROMPT,
    RESET_MODES,
    build_messages_prompt,
    build_token_ids_prompt,
    build_token_ids_retry_prompt,
)


# ──────────────────────────────────────────────────────────────────────────
# Tokenizer fixtures
# ──────────────────────────────────────────────────────────────────────────

# Each entry: (test id, repo id, {"bos": str, "eos_turn": str, "end_of_text": str})
# ``eos_turn``    — end-of-turn marker (closes a single turn).
# ``end_of_text`` — end-of-sequence marker (the expected boundary token for
# token_boundary mode). For tokenizers that don't have a distinct EOS separate
# from end-of-turn (e.g. many non-llama families), this may equal ``eos_turn``.
TOKENIZER_SPECS: list[tuple[str, str, dict[str, str]]] = [
    (
        "llama31_8b_instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        {
            "bos": "<|begin_of_text|>",
            "eos_turn": "<|eot_id|>",
            "end_of_text": "<|end_of_text|>",
        },
    ),
]


def _load_tokenizer(repo_id: str):
    try:
        from transformers import AutoTokenizer
    except ImportError:
        pytest.skip("transformers not installed")
    try:
        return AutoTokenizer.from_pretrained(repo_id, use_fast=True)
    except Exception as exc:  # network / auth failure; cacheable
        pytest.skip(f"Tokenizer {repo_id!r} unavailable: {exc}")


@pytest.fixture(
    params=TOKENIZER_SPECS,
    ids=[spec[0] for spec in TOKENIZER_SPECS],
)
def tokenizer_spec(request) -> dict[str, Any]:
    name, repo_id, specials = request.param
    tok = _load_tokenizer(repo_id)
    return {"name": name, "repo_id": repo_id, "tokenizer": tok, "specials": specials}


# ──────────────────────────────────────────────────────────────────────────
# Messages-level builder (reset_mode ∈ {none, soft})
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture
def rollout_messages() -> list[dict[str, str]]:
    return [
        {"role": "user", "content": "Hi."},
        {"role": "assistant", "content": "Hello! How can I help?"},
        {"role": "user", "content": "Tell me about yourself."},
        {"role": "assistant", "content": "I'm an assistant."},
    ]


ITEM_USER_CONTENT = 'On a scale of 1 to 5, rate: "I am an outgoing person."'


def test_reset_modes_constant_tuple():
    assert set(RESET_MODES) == {"none", "soft", "token_boundary"}


def test_none_mode_matches_legacy_layout(rollout_messages):
    prompt = build_messages_prompt(
        rollout_messages, ITEM_USER_CONTENT, reset_mode="none"
    )
    # Rollout passed through verbatim, then exactly one user turn appended.
    assert prompt.messages[:-1] == list(rollout_messages)
    assert prompt.messages[-1] == {"role": "user", "content": ITEM_USER_CONTENT}


def test_none_mode_with_trait_mcq_prefill_adds_assistant_turn(rollout_messages):
    prompt = build_messages_prompt(
        rollout_messages,
        ITEM_USER_CONTENT,
        reset_mode="none",
        trait_mcq_prefill="Answer ",
    )
    assert prompt.messages[-2] == {"role": "user", "content": ITEM_USER_CONTENT}
    assert prompt.messages[-1] == {"role": "assistant", "content": "Answer "}


def test_soft_mode_inserts_system_between_rollout_and_item(rollout_messages):
    prompt = build_messages_prompt(
        rollout_messages, ITEM_USER_CONTENT, reset_mode="soft"
    )
    # Rollout unchanged.
    assert prompt.messages[: len(rollout_messages)] == list(rollout_messages)
    # Reset system message immediately after rollout.
    reset_msg = prompt.messages[len(rollout_messages)]
    assert reset_msg["role"] == "system"
    assert reset_msg["content"] == DEFAULT_SOFT_RESET_SYSTEM_PROMPT
    # Item follows the reset.
    assert prompt.messages[len(rollout_messages) + 1] == {
        "role": "user",
        "content": ITEM_USER_CONTENT,
    }


def test_soft_mode_respects_custom_system_prompt(rollout_messages):
    custom = "New topic; ignore everything above."
    prompt = build_messages_prompt(
        rollout_messages,
        ITEM_USER_CONTENT,
        reset_mode="soft",
        soft_reset_system_prompt=custom,
    )
    reset_msg = prompt.messages[len(rollout_messages)]
    assert reset_msg == {"role": "system", "content": custom}


def test_build_messages_rejects_token_boundary(rollout_messages):
    with pytest.raises(ValueError, match="token_boundary"):
        build_messages_prompt(
            rollout_messages, ITEM_USER_CONTENT, reset_mode="token_boundary"
        )


def test_build_messages_rejects_unknown_mode(rollout_messages):
    with pytest.raises(ValueError, match="Unknown reset_mode"):
        build_messages_prompt(
            rollout_messages, ITEM_USER_CONTENT, reset_mode="bogus"
        )


def test_builder_does_not_mutate_inputs(rollout_messages):
    original = [dict(m) for m in rollout_messages]
    build_messages_prompt(rollout_messages, ITEM_USER_CONTENT, reset_mode="soft")
    assert rollout_messages == original


# ──────────────────────────────────────────────────────────────────────────
# Token-boundary builder (reset_mode == token_boundary)
# ──────────────────────────────────────────────────────────────────────────


def _special_id(tok, s: str) -> int:
    tid = tok.convert_tokens_to_ids(s)
    assert isinstance(tid, int) and tid != getattr(tok, "unk_token_id", -1), (
        f"Tokenizer {tok.__class__.__name__} does not recognise {s!r} as a "
        f"special token"
    )
    return tid


def test_token_boundary_starts_with_bos_and_has_single_boundary(
    tokenizer_spec, rollout_messages
):
    tok = tokenizer_spec["tokenizer"]
    specials = tokenizer_spec["specials"]
    bos_id = _special_id(tok, specials["bos"])
    boundary_str = specials["end_of_text"]
    boundary_id = _special_id(tok, boundary_str)

    prompt = build_token_ids_prompt(
        tok,
        rollout_messages,
        ITEM_USER_CONTENT,
        boundary_token=boundary_str,
    )
    ids = prompt.token_ids

    # Starts with exactly one BOS.
    assert ids[0] == bos_id, "First token must be BOS"
    assert ids[1] != bos_id, "Second token must not duplicate BOS"

    # Exactly two BOS tokens overall: one at start, one at start of fresh chat.
    assert ids.count(bos_id) == 2, (
        f"Expected exactly two BOS tokens (one at sequence start, one after "
        f"boundary); got {ids.count(bos_id)}"
    )

    # Boundary token appears exactly once, and sits immediately before the
    # second BOS.
    assert ids.count(boundary_id) == 1
    b_idx = ids.index(boundary_id)
    assert ids[b_idx + 1] == bos_id, (
        "Token immediately after boundary must be the fresh-chat BOS"
    )
    # Boundary sits at the recorded split.
    assert b_idx == prompt.num_rollout_tokens
    assert prompt.boundary_token_ids == [boundary_id]


def test_token_boundary_preserves_item_text_after_boundary(
    tokenizer_spec, rollout_messages
):
    tok = tokenizer_spec["tokenizer"]
    specials = tokenizer_spec["specials"]
    prompt = build_token_ids_prompt(
        tok,
        rollout_messages,
        ITEM_USER_CONTENT,
        boundary_token=specials["end_of_text"],
    )
    boundary_id = _special_id(tok, specials["end_of_text"])
    b_idx = prompt.token_ids.index(boundary_id)
    fresh_decoded = tok.decode(prompt.token_ids[b_idx + 1 :])
    assert ITEM_USER_CONTENT in fresh_decoded, (
        f"Questionnaire item text not found in decoded fresh-chat segment. "
        f"Fresh segment: {fresh_decoded!r}"
    )


def test_token_boundary_preserves_rollout_content_before_boundary(
    tokenizer_spec, rollout_messages
):
    tok = tokenizer_spec["tokenizer"]
    specials = tokenizer_spec["specials"]
    prompt = build_token_ids_prompt(
        tok,
        rollout_messages,
        ITEM_USER_CONTENT,
        boundary_token=specials["end_of_text"],
    )
    rollout_decoded = tok.decode(
        prompt.token_ids[: prompt.num_rollout_tokens]
    )
    # Every rollout user / assistant content string must round-trip.
    for msg in rollout_messages:
        assert msg["content"] in rollout_decoded, (
            f"Rollout content {msg['content']!r} missing from decoded rollout "
            f"segment"
        )
    # And the rollout segment must NOT contain the questionnaire item text —
    # that belongs to the post-boundary side only.
    assert ITEM_USER_CONTENT not in rollout_decoded


def test_token_boundary_rollout_ends_with_turn_terminator(
    tokenizer_spec, rollout_messages
):
    """Rollout side should end with <|eot_id|> (no fresh generation prompt)."""
    tok = tokenizer_spec["tokenizer"]
    specials = tokenizer_spec["specials"]
    prompt = build_token_ids_prompt(
        tok,
        rollout_messages,
        ITEM_USER_CONTENT,
        boundary_token=specials["end_of_text"],
    )
    eot_id = _special_id(tok, specials["eos_turn"])
    # The last rollout token should be the turn terminator.
    last_rollout_token = prompt.token_ids[prompt.num_rollout_tokens - 1]
    assert last_rollout_token == eot_id, (
        f"Rollout segment must end with {specials['eos_turn']!r}; ended with "
        f"token id {last_rollout_token}"
    )


def test_token_boundary_trait_mcq_prefill_lands_at_end(
    tokenizer_spec, rollout_messages
):
    tok = tokenizer_spec["tokenizer"]
    specials = tokenizer_spec["specials"]
    prompt = build_token_ids_prompt(
        tok,
        rollout_messages,
        ITEM_USER_CONTENT,
        boundary_token=specials["end_of_text"],
        trait_mcq_prefill="Answer ",
    )
    # The final decoded slice must end with "Answer" (possibly with the
    # chat-template stripping the trailing space — we allow either).
    tail = tok.decode(prompt.token_ids[-6:])
    assert tail.rstrip().endswith("Answer"), (
        f"Prefill 'Answer' not at tail; got tail={tail!r}"
    )
    # No trailing <|eot_id|> — model should continue from prefill, not start
    # a new assistant turn.
    eot_id = _special_id(tok, specials["eos_turn"])
    assert prompt.token_ids[-1] != eot_id


def test_token_boundary_integer_boundary_token(tokenizer_spec, rollout_messages):
    tok = tokenizer_spec["tokenizer"]
    specials = tokenizer_spec["specials"]
    expected_id = _special_id(tok, specials["end_of_text"])
    prompt = build_token_ids_prompt(
        tok,
        rollout_messages,
        ITEM_USER_CONTENT,
        boundary_token=expected_id,
    )
    assert prompt.boundary_token_ids == [expected_id]
    assert prompt.token_ids[prompt.num_rollout_tokens] == expected_id


def test_token_boundary_custom_multi_token_boundary(tokenizer_spec, rollout_messages):
    """Custom multi-token boundary sequences are passed through as-is."""
    tok = tokenizer_spec["tokenizer"]
    specials = tokenizer_spec["specials"]
    eot_id = _special_id(tok, specials["eos_turn"])
    eos_id = _special_id(tok, specials["end_of_text"])
    prompt = build_token_ids_prompt(
        tok,
        rollout_messages,
        ITEM_USER_CONTENT,
        boundary_token=[eot_id, eos_id],
    )
    assert prompt.boundary_token_ids == [eot_id, eos_id]
    assert prompt.token_ids[
        prompt.num_rollout_tokens : prompt.num_rollout_tokens + 2
    ] == [eot_id, eos_id]


def test_token_boundary_unknown_boundary_token_raises(tokenizer_spec, rollout_messages):
    tok = tokenizer_spec["tokenizer"]
    with pytest.raises(ValueError, match="not a known special token"):
        build_token_ids_prompt(
            tok,
            rollout_messages,
            ITEM_USER_CONTENT,
            boundary_token="<|totally_not_a_real_special_token|>",
        )


# ──────────────────────────────────────────────────────────────────────────
# Retry prompt (token_boundary)
# ──────────────────────────────────────────────────────────────────────────


def test_retry_preserves_rollout_and_boundary(tokenizer_spec, rollout_messages):
    tok = tokenizer_spec["tokenizer"]
    specials = tokenizer_spec["specials"]
    baseline = build_token_ids_prompt(
        tok,
        rollout_messages,
        ITEM_USER_CONTENT,
        boundary_token=specials["end_of_text"],
    )
    retry = build_token_ids_retry_prompt(
        tok,
        rollout_messages,
        ITEM_USER_CONTENT,
        prior_assistant_text="blah blah",
        retry_user_content="Please answer properly.",
        boundary_token=specials["end_of_text"],
    )
    # The rollout + boundary prefix is byte-identical to the baseline build.
    prefix_len = baseline.num_rollout_tokens + len(baseline.boundary_token_ids)
    assert retry.token_ids[:prefix_len] == baseline.token_ids[:prefix_len]
    assert retry.num_rollout_tokens == baseline.num_rollout_tokens
    assert retry.boundary_token_ids == baseline.boundary_token_ids


def test_retry_includes_prior_and_retry_text(tokenizer_spec, rollout_messages):
    tok = tokenizer_spec["tokenizer"]
    specials = tokenizer_spec["specials"]
    retry = build_token_ids_retry_prompt(
        tok,
        rollout_messages,
        ITEM_USER_CONTENT,
        prior_assistant_text="I don't want to say.",
        retry_user_content="Please respond with only a number.",
        boundary_token=specials["end_of_text"],
    )
    fresh_start = retry.num_rollout_tokens + len(retry.boundary_token_ids)
    fresh_decoded = tok.decode(retry.token_ids[fresh_start:])
    assert ITEM_USER_CONTENT in fresh_decoded
    assert "I don't want to say." in fresh_decoded
    assert "Please respond with only a number." in fresh_decoded


def test_retry_trait_mcq_reconstructs_prior_full_answer(
    tokenizer_spec, rollout_messages
):
    tok = tokenizer_spec["tokenizer"]
    specials = tokenizer_spec["specials"]
    retry = build_token_ids_retry_prompt(
        tok,
        rollout_messages,
        ITEM_USER_CONTENT,
        prior_assistant_text="maybe C, I guess",
        retry_user_content="Please respond with only 'A', 'B', 'C', or 'D'.",
        boundary_token=specials["end_of_text"],
        trait_mcq_prefill="Answer ",
    )
    fresh_start = retry.num_rollout_tokens + len(retry.boundary_token_ids)
    fresh_decoded = tok.decode(retry.token_ids[fresh_start:])
    # Full prior assistant turn should read as prefill + response.
    assert "Answer maybe C, I guess" in fresh_decoded
    # Ends with the new prefill (and NOT a fresh eot_id).
    tail = tok.decode(retry.token_ids[-6:])
    assert tail.rstrip().endswith("Answer")
    eot_id = _special_id(tok, specials["eos_turn"])
    assert retry.token_ids[-1] != eot_id
