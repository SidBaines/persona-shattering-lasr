"""Encode parsed questionnaire responses into the FA response matrix.

Per column type:
    FC         — +1 if choice == "A" else -1
    fc_pair    — +1 if choice is the per-item high pole, else -1
    Vignette   — per-dimension scores from the chosen option's ``scoring`` dict
    Likert     — integer 1–5, reverse-keyed per item
    trait_mcq  — trait-aligned [0, 1] score from per-item ``answer_mapping``;
                 soft expectation Σ P(letter)·mapping[letter] when
                 ``choice_probs`` is supplied, else hard argmax lookup.

The encoding version is bumped when these semantics change so cached response
matrices produced under an older encoding are rebuilt from raw_responses.jsonl
(the authoritative source of truth) rather than re-used blindly.

History:
    v1 — trait_mcq = A=1..D=4 letter-integer.
    v2 — trait_mcq = answer_mapping-based 0/1 trait-aligned (soft: Σ P·mapping).
"""

from __future__ import annotations

import json

import numpy as np


RESPONSE_MATRIX_ENCODING_VERSION = 2


def fill_matrix_from_choice(
    response_matrix: np.ndarray,
    k: int,
    item_id: str,
    choice: str | int,
    item_to_cols: dict[str, list[tuple[int, str | None]]],
    vig_scoring: dict[str, dict[str, dict[str, int]]],
    likert_reverse: dict[str, bool],
    trait_mcq_mapping: dict[str, dict[str, int]] | None = None,
    fc_pair_high: dict[str, str] | None = None,
    choice_probs: dict[str, float] | None = None,
) -> None:
    """Fill matrix columns for persona k given their choice on item_id.

    Column type is inferred from the column definition and the item-id set:
    - FC:        single column with dimension=None, encoded +1=A / -1=B
    - fc_pair:   single column with dimension=axis, encoded +1=high / -1=low
                 (aligned to per-item ``high_option`` so axis polarity is
                 consistent across items regardless of A/B counterbalancing)
    - Vignette:  multiple columns (one per dimension) via option scoring dict
    - Likert:    single column with dimension set, encoded 1-5 with optional
                 reversal
    - trait_mcq: single column with dimension set, encoded as the trait-aligned
                 score in [0,1]. With ``choice_probs`` we compute the soft
                 expectation Σ P(letter)·answer_mapping[letter]; otherwise we
                 hard-score the argmax letter via answer_mapping[choice]. The
                 per-item ``answer_mapping`` MUST be supplied for any item_id
                 in ``trait_mcq_mapping`` — a missing mapping is a hard error
                 because without it we cannot orient the column to the trait
                 pole and any resulting factor analysis would be meaningless.
    """
    cols = item_to_cols.get(item_id, [])
    if not cols:
        return

    col_idx_0, dim_0 = cols[0]

    if trait_mcq_mapping is not None and item_id in trait_mcq_mapping:
        mapping = trait_mcq_mapping[item_id]
        if not mapping:
            raise ValueError(
                f"trait_mcq item {item_id!r} has empty answer_mapping — refusing to "
                "fall back to letter-integer encoding, which would orient factors "
                "arbitrarily. Fix the questionnaire file to include a valid "
                "answer_mapping."
            )
        if choice_probs:
            # Soft trait-aligned score: Σ P(letter) · mapping[letter]
            total = 0.0
            covered = 0.0
            for letter, p in choice_probs.items():
                if str(letter) in mapping:
                    total += float(p) * float(mapping[str(letter)])
                    covered += float(p)
            if covered > 0:
                response_matrix[k, col_idx_0] = float(total)
        else:
            if isinstance(choice, str) and choice in mapping:
                response_matrix[k, col_idx_0] = float(mapping[choice])
        return

    if fc_pair_high is not None and item_id in fc_pair_high:
        # fc_pair: +1 if the chosen letter is the high pole of the axis, else -1.
        high = fc_pair_high[item_id]
        if isinstance(choice, str) and choice in ("A", "B"):
            response_matrix[k, col_idx_0] = 1.0 if choice == high else -1.0
        return

    if dim_0 is None:
        # FC: one column, +1=A, -1=B
        response_matrix[k, col_idx_0] = 1.0 if choice == "A" else -1.0
    elif isinstance(choice, str):
        # Vignette: fill per-dimension scores from chosen option's scoring dict
        option_scores = vig_scoring.get(item_id, {}).get(choice, {})
        for col_idx, dim in cols:
            response_matrix[k, col_idx] = float(option_scores.get(dim, 0))
    else:
        # Likert: 1–5, apply reverse keying.
        # When choice_probs is supplied (logprob-scored path), compute
        # the soft expected value Σ i · P(i) over digit keys — gives a
        # continuous float in [1, scale] instead of the argmax integer.
        # Reverse-keyed items use (scale+1) − score so the polarity
        # convention matches the hard-scored path. Falls through to
        # argmax encoding when the caller didn't provide digit probs.
        if choice_probs and all(
            isinstance(k_, (int, float)) or (isinstance(k_, str) and k_.isdigit())
            for k_ in choice_probs.keys()
        ):
            numeric_probs = {int(k_): float(v) for k_, v in choice_probs.items()}
            if numeric_probs:
                score = sum(i * p for i, p in numeric_probs.items())
                scale = max(numeric_probs.keys())
                if likert_reverse.get(item_id, False):
                    score = (scale + 1) - score
                response_matrix[k, col_idx_0] = float(score)
                return
        score = int(choice)
        if likert_reverse.get(item_id, False):
            score = 6 - score
        response_matrix[k, col_idx_0] = float(score)


def record_response(
    response_matrix: np.ndarray,
    k: int,
    item: dict,
    choice: str | int,
    raw_text: str,
    item_to_cols: dict[str, list[tuple[int, str | None]]],
    vig_scoring: dict[str, dict[str, dict[str, int]]],
    likert_reverse: dict[str, bool],
    log_fh,
    trait_mcq_mapping: dict[str, dict[str, int]] | None = None,
    fc_pair_high: dict[str, str] | None = None,
    choice_probs: dict[str, float] | None = None,
) -> None:
    """Fill matrix and log raw response to an open file handle."""
    item_id = item["id"]
    fill_matrix_from_choice(
        response_matrix,
        k,
        item_id,
        choice,
        item_to_cols,
        vig_scoring,
        likert_reverse,
        trait_mcq_mapping=trait_mcq_mapping,
        fc_pair_high=fc_pair_high,
        choice_probs=choice_probs,
    )
    log_fh.write(
        json.dumps(
            {
                "k": k,
                "item_id": item_id,
                "item_type": item["type"],
                "parsed_choice": choice,
                "raw": raw_text,
            },
            ensure_ascii=False,
        )
        + "\n"
    )
