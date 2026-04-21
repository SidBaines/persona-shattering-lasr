"""Parsers for free-text LLM responses to psychometric items.

Each questionnaire item expects a specific answer shape (Likert integer 1–5,
forced-choice A/B, vignette/trait_mcq ABCD). These parsers recover the
answer from an LLM's raw text, and — for logprob-scored trait_mcq / fc_pair
items — convert the top-k logprob dict into a per-letter probability
distribution plus the total choice-letter mass.
"""

from __future__ import annotations

import math
import re


# Tokenizer-variant letter sets for logprob extraction.
# Different tokenizers encode a bare letter as "A", "▁A", "ĠA", " A", "a".
TRAIT_MCQ_LETTER_VARIANTS: dict[str, set[str]] = {
    letter: {
        letter,
        f"▁{letter}",
        f"Ġ{letter}",
        f" {letter}",
        letter.lower(),
        f" {letter.lower()}",
    }
    for letter in "ABCD"
}


def _digit_alias_variants(digit: int) -> set[str]:
    """Token-variants of an option-ordinal digit (e.g. '1', '▁1', '1.', ')1')."""
    d = str(digit)
    return {
        d,
        f"▁{d}",
        f"Ġ{d}",
        f" {d}",
        # Models often emit "1." or "1)" when listing answer numbers.
        f"{d}.",
        f"{d})",
        f" {d}.",
        f" {d})",
    }


def parse_likert_response(text: str) -> int | None:
    """Parse a Likert-scale integer (1-5) from an LLM response.

    Handles common patterns:
      - Clean digit: "3"
      - Digit with explanation: "3 - I somewhat agree"
      - Verbose: "I would rate this a 3"
      - Fallback: first digit 1-5 found in text
    """
    text = text.strip()
    if not text:
        return None

    # Pattern 1: starts with a digit 1-5 (possibly wrapped in markdown bold/italic)
    m = re.match(r"^[\s*_]*([1-5])\b", text)
    if m:
        return int(m.group(1))

    # Pattern 2: find a digit 1-5 in common phrasing patterns
    m = re.search(
        r"\b(?:rate|rating|score|answer|respond|give)\b.*?([1-5])\b",
        text,
        re.IGNORECASE,
    )
    if m:
        return int(m.group(1))

    # Pattern 3: fallback — find first standalone digit 1-5
    m = re.search(r"\b([1-5])\b", text)
    if m:
        return int(m.group(1))

    return None


def parse_ab_response(text: str) -> str | None:
    """Parse a forced-choice A/B response. Returns 'A' or 'B', or None."""
    text = text.strip()
    if not text:
        return None
    # Check start of response first (most common pattern)
    m = re.match(r"^[\s*_]*([AB])\b", text.upper())
    if m:
        return m.group(1)
    # Fallback: first standalone A or B
    m = re.search(r"\b([AB])\b", text.upper())
    return m.group(1) if m else None


def parse_abcd_response(text: str) -> str | None:
    """Parse a vignette A/B/C/D response. Returns 'A'–'D', or None."""
    text = text.strip()
    if not text:
        return None
    m = re.match(r"^[\s*_]*([ABCD])\b", text.upper())
    if m:
        return m.group(1)
    m = re.search(r"\b([ABCD])\b", text.upper())
    return m.group(1) if m else None


def parse_item_response(item: dict, text: str) -> str | int | None:
    """Parse the raw LLM response for any item type.

    Returns:
        'A'/'B' for forced_choice/fc_pair, 'A'–'D' for vignette/trait_mcq,
        int 1-5 for likert, or None on parse failure.
    """
    if item["type"] in ("forced_choice", "fc_pair"):
        return parse_ab_response(text)
    elif item["type"] in ("vignette", "trait_mcq"):
        return parse_abcd_response(text)
    else:
        return parse_likert_response(text)


def parse_top_logprobs_to_likert_probs(
    top_logprobs: dict[str, float],
    scale: int = 5,
) -> tuple[dict[int, float], float]:
    """Extract per-digit probabilities for a Likert item from top-k logprobs.

    Mirrors :func:`parse_top_logprobs_to_choice_probs` but for integer
    Likert answers (1..scale). Used when the questionnaire is configured
    with ``use_logprobs=True`` and the "aside" phrasing (prefilled with
    ``"Answer: "``) so the first generated token is a digit.

    Args:
        top_logprobs: Mapping ``decoded_token -> logprob`` for the first
            generated token.
        scale: Max Likert value (default 5, i.e. 1/2/3/4/5).

    Returns:
        ``(probs, choice_mass)`` where ``probs`` maps each integer
        ``1..scale`` to its softmax-normalised probability over the
        found digits only, and ``choice_mass`` is the sum of
        ``exp(lp)`` across those digits out of the full vocabulary.
    """
    digits = [str(d) for d in range(1, scale + 1)]
    found: dict[int, float] = {}
    for d_str in digits:
        variants = {
            d_str,
            f"▁{d_str}",
            f"Ġ{d_str}",
            f" {d_str}",
        }
        for tok, lp in top_logprobs.items():
            if tok in variants:
                found[int(d_str)] = float(lp)
                break
    if not found:
        return {}, 0.0

    choice_mass = sum(math.exp(lp) for lp in found.values())
    max_lp = max(found.values())
    exp_vals = {k: math.exp(v - max_lp) for k, v in found.items()}
    total = sum(exp_vals.values())
    probs = {k: v / total for k, v in exp_vals.items()}
    return probs, float(choice_mass)


def parse_top_logprobs_to_choice_probs(
    top_logprobs: dict[str, float],
    num_choices: int = 4,
    *,
    include_digit_aliases: bool = False,
) -> tuple[dict[str, float], float]:
    """Extract per-letter probabilities from a top-k logprob dict.

    Handles common tokenizer variants (``A``, ``▁A``, ``ĠA``, `` A``, ``a``).

    Args:
        top_logprobs: Mapping ``decoded_token -> logprob`` for the first
            generated token.
        num_choices: Number of answer choices (default 4, i.e. A/B/C/D).
        include_digit_aliases: When True, also treats position-ordinal
            digit tokens (``1`` → A, ``2`` → B, ``3`` → C, ``4`` → D for
            num_choices=4) as equivalent evidence for the corresponding
            letter, combining their logprob mass. Needed for models that
            answer multiple-choice questions with option ordinals
            (Mistral-7B, Zephyr-7B) rather than letters. When a letter
            and its digit alias both appear in top-k, their
            probabilities are summed in linear (exp) space before
            re-logging; the result is the logprob of "either the letter
            OR its digit appearing", which is what we want.

    Returns:
        ``(probs, choice_mass)`` where:
          - ``probs`` is a dict mapping each found letter to its softmax-
            normalized probability over the found letters only.
          - ``choice_mass`` is the total probability mass on choice tokens
            (letters, and digit aliases if enabled) out of the full
            vocabulary.
        Both are empty / 0.0 when no choice letter/digit appears in top-k.
    """
    letters = [chr(ord("A") + i) for i in range(num_choices)]
    found: dict[str, float] = {}
    for i, letter in enumerate(letters):
        variants = set(TRAIT_MCQ_LETTER_VARIANTS.get(letter) or {
            letter,
            f"▁{letter}",
            f"Ġ{letter}",
            f" {letter}",
            letter.lower(),
        })
        if include_digit_aliases:
            variants |= _digit_alias_variants(i + 1)
        # For a given letter, any one of its variants matches. Some
        # tokenizers may emit both forms in the top-k (letter + digit);
        # when that happens, sum their linear probabilities.
        matching_lps: list[float] = [
            float(lp) for tok, lp in top_logprobs.items() if tok in variants
        ]
        if matching_lps:
            combined_prob = sum(math.exp(lp) for lp in matching_lps)
            found[letter] = math.log(combined_prob)

    if not found:
        return {}, 0.0

    choice_mass = sum(math.exp(lp) for lp in found.values())
    max_lp = max(found.values())
    exp_vals = {k: math.exp(v - max_lp) for k, v in found.items()}
    total = sum(exp_vals.values())
    probs = {k: v / total for k, v in exp_vals.items()}
    return probs, float(choice_mass)
