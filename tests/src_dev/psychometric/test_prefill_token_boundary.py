"""Regression test for the BPE-boundary prefill bug.

Background — the bug we're guarding against:

* ``TRAIT_MCQ_PREFILL`` used to be ``"Answer "`` (trailing space).
* Qwen2.5's and Llama-3.1's BPE tokenizers merge space+letter into one
  token (``" A" == token 362`` on Qwen, similar on Llama), rather than
  ``[' ', 'A']``.
* So a prefill ending in a standalone whitespace token places the model
  in a state where the tokenizer-natural letter continuation (``" A"``,
  one token) decodes as ``"Answer  A"`` — double space, out of
  distribution. Qwen2.5 disperses ~12% of top-20 mass onto escape routes
  like ``" Answer"`` / ``<|im_end|>`` rather than committing to a
  letter, which tanks per-cell ``choice_mass`` and, in turn, the
  soft-expectation scores used for FA.
* Fix: ``TRAIT_MCQ_PREFILL = "Answer"`` (no trailing space). Letter-mass
  share recovers to ~100%. See
  ``scripts_dev/psychometric_assessment/prefill_ablation.py`` for the
  full investigation.

The invariant this test enforces:

    Tokenizing the prefill must NOT leave a standalone-whitespace token
    at the end. If it does, a space-prefixed continuation will produce
    a double-space decoded string.

The check is tokenizer-specific; we test against Qwen2.5-7B-Instruct
(public, ungated). Llama-3.1-8B-Instruct would be the other natural
target but it's gated; the invariant is general to any BPE tokenizer
that merges space+letter.

The test does NOT load the 7B model — only its tokenizer (tiny). It
requires network on first run to pull the tokenizer files; subsequent
runs use the local HF cache.
"""

from __future__ import annotations

import pytest


pytestmark = pytest.mark.skipif(
    pytest.importorskip("transformers", reason="transformers not installed") is None,
    reason="transformers not available",
)


QWEN_TOK_ID = "Qwen/Qwen2.5-7B-Instruct"


def _get_tokenizer():
    """Return the Qwen2.5 tokenizer, skipping if offline / token unavailable."""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(QWEN_TOK_ID)
    except OSError as e:
        pytest.skip(f"Qwen2.5 tokenizer unavailable (offline?): {e}")


def test_trait_mcq_prefill_no_trailing_whitespace_token():
    """``TRAIT_MCQ_PREFILL`` must not tokenize to a trailing whitespace token.

    See module docstring for why. This is the minimal guard — if the
    constant regresses to ``"Answer "`` or ``"Answer:  "`` or anything
    else that ends in a standalone space token, this fails loudly.
    """
    from src_dev.psychometric.item_prompts import TRAIT_MCQ_PREFILL

    tok = _get_tokenizer()
    ids = tok.encode(TRAIT_MCQ_PREFILL, add_special_tokens=False)
    assert ids, f"Prefill {TRAIT_MCQ_PREFILL!r} tokenizes to nothing"
    last_piece = tok.decode([ids[-1]])
    assert last_piece.strip(), (
        f"TRAIT_MCQ_PREFILL = {TRAIT_MCQ_PREFILL!r} ends in a standalone "
        f"whitespace token ({last_piece!r}, id {ids[-1]}). This re-introduces "
        f"the BPE boundary bug — natural space-prefixed letter continuations "
        f"like ' A' (one token) will decode as double-space text and the model "
        f"will disperse mass onto escape tokens instead of committing to a "
        f"letter. Keep the prefill ending at a word-character boundary (e.g. "
        f"'Answer'). See src_dev/psychometric/item_prompts.py and "
        f"scripts_dev/psychometric_assessment/prefill_ablation.py."
    )


def test_trait_mcq_prefill_concatenates_cleanly_with_space_letter():
    """Decoded prefill + continuation must be a natural single-space string.

    For each letter in A/B/C/D, verify that tokenising the prefill and
    appending the natural ``" X"`` token (one token on Qwen2.5 BPE) and
    then decoding yields exactly ``"<prefill> X"`` — no double spaces,
    no dropped characters.
    """
    from src_dev.psychometric.item_prompts import TRAIT_MCQ_PREFILL

    tok = _get_tokenizer()
    prefill_ids = tok.encode(TRAIT_MCQ_PREFILL, add_special_tokens=False)
    for letter in ("A", "B", "C", "D"):
        space_letter_ids = tok.encode(f" {letter}", add_special_tokens=False)
        # Sanity: the natural " X" should be a single BPE token on Qwen2.5.
        # If this assertion ever fails, the tokenizer has changed its
        # space+letter merging and the invariant may need revisiting.
        assert len(space_letter_ids) == 1, (
            f"Qwen2.5 tokenizer merged {' '+letter!r} into "
            f"{len(space_letter_ids)} tokens — unexpected; check tokenizer "
            f"version or update the test expectations."
        )
        combined = tok.decode(prefill_ids + space_letter_ids)
        expected = f"{TRAIT_MCQ_PREFILL} {letter}"
        assert combined == expected, (
            f"Prefill + natural letter continuation produces dirty text.\n"
            f"  prefill           = {TRAIT_MCQ_PREFILL!r}\n"
            f"  prefill ids       = {prefill_ids}\n"
            f"  + space-letter id = {space_letter_ids[0]} ({tok.decode(space_letter_ids)!r})\n"
            f"  decoded           = {combined!r}\n"
            f"  expected          = {expected!r}\n"
            f"Likely cause: prefill ends in a standalone whitespace token — "
            f"creates a double-space state that forces the model off the "
            f"natural letter-emission path. See module docstring."
        )
