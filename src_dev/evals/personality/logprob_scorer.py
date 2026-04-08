"""Logprob-based solver, scorer, and metric for TRAIT personality MCQ evaluation.

Instead of generating text and parsing "ANSWER: X", this approach:
1. Formats the MCQ prompt identically to inspect_ai's multiple_choice solver
2. Optionally appends a forced prefill (e.g. "ANSWER: ") as a partial assistant turn
3. Generates a single token with logprobs enabled
4. Reads the logprobs for choice tokens (A, B, C, D)
5. Softmax-normalizes to get P(A), P(B), P(C), P(D)
6. Computes a continuous 0-1 trait score using the answer mapping

Raw logprobs are stored in Score.metadata for offline re-analysis.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from inspect_ai.model._generate_config import GenerateConfig
from inspect_ai.model._model_output import Logprob, TopLogprob
from inspect_ai.scorer import (
    Metric,
    SampleScore,
    Score,
    Scorer,
    Target,
    Value,
    metric,
    scorer,
)
from inspect_ai.solver import TaskState
from inspect_ai.solver._multiple_choice import (
    SINGLE_ANSWER_TEMPLATE,
    answer_options,
    prompt as format_mcq_prompt,
)
from inspect_ai.solver._solver import Generate, Solver, solver

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Choice letters used in TRAIT (4-choice MCQ).
_CHOICE_LETTERS = ("A", "B", "C", "D")

# Common tokenizer representations of a bare letter token.
# Different tokenizers encode "A" as "A", "▁A", "ĠA", " A", etc.
_LETTER_VARIANTS = {
    letter: {letter, f"▁{letter}", f"Ġ{letter}", f" {letter}", letter.lower()}
    for letter in _CHOICE_LETTERS
}


def _find_choice_logprobs(
    top_logprobs: list[TopLogprob],
) -> dict[str, float]:
    """Extract logprobs for choice letters from a top-k logprob list.

    Returns a dict mapping canonical letter (A/B/C/D) to its logprob.
    If a letter is not in top-k, it gets -inf (excluded from softmax).
    """
    result: dict[str, float] = {}
    for letter, variants in _LETTER_VARIANTS.items():
        for entry in top_logprobs:
            if entry.token in variants:
                result[letter] = float(entry.logprob)
                break
    return result


def _softmax_over_choices(logprobs: dict[str, float]) -> dict[str, float]:
    """Softmax-normalize logprobs for the found choice letters.

    Only normalizes over the letters actually present in the logprobs dict.
    """
    if not logprobs:
        return {}
    max_lp = max(logprobs.values())
    exp_vals = {k: math.exp(v - max_lp) for k, v in logprobs.items()}
    total = sum(exp_vals.values())
    return {k: v / total for k, v in exp_vals.items()}


def _compute_trait_score(
    probs: dict[str, float],
    answer_mapping: dict[str, int],
) -> float:
    """Compute expected trait score from choice probabilities and answer mapping.

    For TRAIT: mapping is {"A": 1, "B": 1, "C": 0, "D": 0}, so this
    returns P(A) + P(B) = P(high trait).
    """
    max_val = max(answer_mapping.values()) if answer_mapping else 1
    score = 0.0
    for letter, prob in probs.items():
        if letter in answer_mapping:
            score += prob * answer_mapping[letter]
    return score / max_val if max_val > 0 else score


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


@solver
def logprob_multiple_choice(
    prefill: str = "ANSWER: ",
    template: str | None = None,
) -> Solver:
    """MCQ solver that uses logprobs instead of text generation.

    Formats the prompt identically to Inspect's multiple_choice(), then
    optionally appends a forced prefill as a partial assistant message so
    the model's first generated token is one of A/B/C/D. Generates with
    max_tokens=1 and logprobs enabled.

    Args:
        prefill: String to prepend as a partial assistant turn. Set to ""
            to disable forced prefill.
        template: MCQ prompt template. Defaults to Inspect's standard
            single-answer template.
    """
    effective_template = template or SINGLE_ANSWER_TEMPLATE

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if not state.choices:
            raise ValueError("logprob_multiple_choice requires samples with choices")

        # Format the user prompt with question + lettered choices.
        state.user_prompt.text = format_mcq_prompt(
            question=state.user_prompt.text,
            choices=state.choices,
            template=effective_template,
        )

        # Optionally inject a partial assistant message as forced prefill.
        if prefill:
            from inspect_ai.model._chat_message import ChatMessageAssistant
            state.messages.append(
                ChatMessageAssistant(
                    content=prefill,
                    model="",
                    source="generate",
                )
            )

        # Generate a single token with logprobs.
        state = await generate(
            state,
            max_tokens=1,
            logprobs=True,
            top_logprobs=20,
        )

        return state

    return solve


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


@scorer(metrics=[])  # Metrics attached at task level.
def logprob_trait_scorer() -> Scorer:
    """Score TRAIT MCQ samples using logprobs.

    Reads the first generated token's top-k logprobs, finds choice letters,
    softmax-normalizes, and computes a continuous 0-1 trait score.

    Raw logprobs and probabilities are stored in Score.metadata for
    offline re-analysis with different normalization schemes.
    """

    async def score(state: TaskState, target: Target) -> Score:
        meta = state.metadata or {}
        answer_mapping = meta.get("answer_mapping", {})

        # Extract logprobs from the first generated token.
        choice = state.output.choices[0] if state.output.choices else None
        logprobs_obj = choice.logprobs if choice else None
        top_lps: list[TopLogprob] = []

        if logprobs_obj and logprobs_obj.content:
            first_token = logprobs_obj.content[0]
            top_lps = first_token.top_logprobs or []

        choice_logprobs = _find_choice_logprobs(top_lps)

        # Compute fraction of probability mass on choice letters (A/B/C/D)
        # out of the full vocabulary.  Since top_lps contains true
        # log-probabilities, exp(lp) is each token's actual probability.
        choice_mass = sum(math.exp(lp) for lp in choice_logprobs.values()) if choice_logprobs else 0.0

        if not choice_logprobs:
            # No choice tokens found in top-k — return NaN score.
            return Score(
                value=float("nan"),
                answer=None,
                explanation="No choice letter tokens found in top logprobs",
                metadata={
                    "logprobs": {},
                    "probs": {},
                    "choice_mass": 0.0,
                    "scoring_method": "logprob",
                },
            )

        probs = _softmax_over_choices(choice_logprobs)
        trait_score = _compute_trait_score(probs, answer_mapping)

        # Best letter by probability (for backward-compat with analysis code).
        best_letter = max(probs, key=probs.get) if probs else None

        return Score(
            value=trait_score,
            answer=best_letter,
            explanation=(
                f"P(high)={trait_score:.4f} | "
                + " ".join(f"P({k})={v:.4f}" for k, v in sorted(probs.items()))
                + f" | mass={choice_mass:.4f}"
            ),
            metadata={
                "logprobs": {k: round(v, 6) for k, v in choice_logprobs.items()},
                "probs": {k: round(v, 6) for k, v in probs.items()},
                "choice_mass": round(choice_mass, 6),
                "scoring_method": "logprob",
            },
        )

    return score


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------


@metric
def logprob_trait_ratio(min_choice_mass: float = 0.0) -> Metric:
    """Per-trait mean of continuous logprob-based trait scores.

    Analogous to the existing ``trait_ratio()`` metric but operates on
    continuous 0-1 scores rather than binary correct/incorrect.

    Args:
        min_choice_mass: Minimum fraction of probability mass on choice
            tokens (A/B/C/D) for a sample to be included.  Samples with
            ``choice_mass < min_choice_mass`` are excluded from the mean.
            Default 0.0 (no filtering).
    """

    def compute(scores: list[SampleScore]) -> Value:
        aggregated: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        n_missing = 0
        n_filtered = 0

        for s in scores:
            meta = s.sample_metadata or {}
            trait = meta.get("trait", "Unknown")

            if s.score is None or not isinstance(s.score.value, (int, float)):
                n_missing += 1
                continue
            val = float(s.score.value)
            if math.isnan(val):
                n_missing += 1
                continue

            # Filter by choice mass if threshold is set.
            if min_choice_mass > 0.0:
                score_meta = s.score.metadata or {}
                cm = score_meta.get("choice_mass", 1.0)
                if isinstance(cm, (int, float)) and cm < min_choice_mass:
                    n_filtered += 1
                    continue

            aggregated[trait] += val
            counts[trait] += 1

        result: dict[str, float] = {}
        for trait in counts:
            result[trait] = aggregated[trait] / counts[trait]
        if n_missing:
            result["_missing"] = float(n_missing)
        if n_filtered:
            result["_filtered_low_mass"] = float(n_filtered)
        return result

    return compute
