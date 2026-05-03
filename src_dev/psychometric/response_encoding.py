"""Encode parsed questionnaire responses into the FA response matrix.

Per column type:
    FC         — +1 if choice == "A" else -1
    fc_pair    — soft 2·P(high) − 1 in [−1, +1] when ``choice_probs`` is
                 supplied (logprob-scored path); hard ±1 from argmax letter
                 otherwise. The soft encoding preserves confidence
                 information that the hard argmax discards (a near-50/50
                 response and a confident high-pole response no longer map
                 to the same cell value), so FA on logprob-scored fc_pair
                 columns sees gradient signal rather than just sign.
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
    v3 — logprob-scored Likert reverse-keying uses the nominal scale
         (``likert_scale``) instead of ``max(digits present in top-k)``.
         Under v2 a strongly-polarised response whose top-k logprobs did
         not span the full 1..5 scale (e.g. only digits {1, 2} present)
         produced a reversed score that was not the true complement —
         breaking the polarity convention for reverse-keyed items. v3
         passes the configured scale through so the reversal is
         ``(likert_scale + 1) − score`` regardless of which digits
         appeared in the top-k.
    (note: ``trait_mcq_encoding`` is a per-run config dial, NOT an
    encoding-version bump — soft_ev and logit produce different cell
    values but share the same "format" contract, and cache collision is
    avoided by appending ``-enc_logit`` to the run-id tag for logit runs
    in ``_questionnaire_run_id`` so existing soft_ev caches remain valid.)
    v4 — ``QuestionnaireStageConfig.min_choice_mass`` is now actually
         applied during matrix encoding. Cells where the top-k logprobs
         carry less than ``min_choice_mass`` total probability on the
         choice tokens (digits for Likert, letters for trait_mcq /
         fc_pair) are recorded as NaN rather than a meaningless "soft
         expectation" over a handful of noise-level probabilities. The
         config field existed before v4 but was silently ignored in
         ``questionnaire_inference`` (it was only wired into the
         separate ``trait_scoring`` stage). v4 applies the gate on both
         the live inference path and the rebuild-from-raw path so
         cached matrices produced under a different threshold are
         automatically re-filtered. Bumped to force one rebuild; future
         threshold changes require manual cache invalidation.
    v5 — fc_pair encoding is now soft 2·P(high) − 1 in [−1, +1] under
         the logprob-scored path (when ``choice_probs`` is supplied).
         Previously fc_pair was hard ±1 from the argmax letter even on
         the logprob-scored path, throwing away the confidence
         information that trait_mcq and Likert both retain via their
         soft-expectation encodings. The soft encoding restores that
         information for FA: a near-50/50 cell and a confident
         high-pole cell now contribute different signal rather than
         the same ±1. The hard-argmax fallback (when no choice_probs)
         is unchanged, so non-logprob runs are bit-for-bit identical.
         The rebuild-from-raw path picks up the new encoding for free
         because raw_responses.jsonl carries the per-cell choice_probs.
"""

from __future__ import annotations

import json

import numpy as np


RESPONSE_MATRIX_ENCODING_VERSION = 5


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
    likert_scale: int = 5,
    trait_mcq_encoding: str = "soft_ev",
    trait_mcq_logit_epsilon: float = 0.005,
) -> None:
    """Fill matrix columns for persona k given their choice on item_id.

    Column type is inferred from the column definition and the item-id set:
    - FC:        single column with dimension=None, encoded +1=A / -1=B
    - fc_pair:   single column with dimension=axis. Logprob-scored path
                 (``choice_probs`` supplied) uses soft 2·P(high) − 1 in
                 [−1, +1]; hard-argmax fallback uses ±1 (high pole / low
                 pole). The per-item ``high_option`` aligns axis polarity
                 across items regardless of which letter corresponds to
                 the high pole, so A/B counterbalancing at conversion
                 time is invisible to the FA.
    - Vignette:  multiple columns (one per dimension) via option scoring dict
    - Likert:    single column with dimension set, encoded 1-5 with optional
                 reversal
    - trait_mcq: single column with dimension set, encoded per
                 ``trait_mcq_encoding`` (default ``"soft_ev"``):
                   - ``"soft_ev"`` — trait-aligned score in [0, 1]. With
                     ``choice_probs`` we compute the soft expectation
                     Σ P(letter)·answer_mapping[letter]; otherwise we
                     hard-score the argmax letter via
                     answer_mapping[choice].
                   - ``"logit"`` — log-odds log(P(high) / (1 − P(high))) in
                     (−∞, +∞), clipped so that P(high) ∈ [ε, 1 − ε] (ε =
                     ``trait_mcq_logit_epsilon``). The logit is the
                     natural parameter of a Bernoulli and the latent
                     linear predictor in a 2PL IRT model, so it gives
                     Pearson-on-logit the right scale properties for FA's
                     linear-Gaussian assumption. Differences at the 0.95
                     → 0.99 end of the P scale (which Pearson-on-P
                     treats as noise) become first-class signal. Only
                     applied when ``choice_probs`` is present — hard
                     argmax responses fall back to ``"soft_ev"`` (0/1)
                     since the logit would be ±∞ on a pure {0,1}
                     observation with no smoothing.

                 The per-item ``answer_mapping`` MUST be supplied for any
                 item_id in ``trait_mcq_mapping`` — a missing mapping is a
                 hard error because without it we cannot orient the column
                 to the trait pole and any resulting factor analysis would
                 be meaningless.
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
            # Accumulate both the original soft_ev sum (Σ P·mapping, which
            # can handle non-binary mappings) and a dedicated p_high for
            # the logit branch (which assumes binary {0,1} poles). Only
            # one of the two is consumed per call depending on
            # ``trait_mcq_encoding``.
            total = 0.0
            p_high = 0.0
            covered = 0.0
            for letter, p in choice_probs.items():
                if str(letter) in mapping:
                    m = float(mapping[str(letter)])
                    p_f = float(p)
                    total += p_f * m
                    if m == 1.0:
                        p_high += p_f
                    covered += p_f
            if covered > 0:
                if trait_mcq_encoding == "logit":
                    p_norm = p_high / covered
                    # Clip to avoid ±∞ on confidently-one-sided cells.
                    # eps = 0.005 → latent range ≈ [−5.3, +5.3].
                    eps = float(trait_mcq_logit_epsilon)
                    p_norm = min(max(p_norm, eps), 1.0 - eps)
                    response_matrix[k, col_idx_0] = float(
                        np.log(p_norm / (1.0 - p_norm))
                    )
                else:
                    # Preserves pre-logit behaviour bit-for-bit.
                    response_matrix[k, col_idx_0] = float(total)
        else:
            # Hard-argmax fallback: emit the 0/1 trait-alignment regardless
            # of encoding choice. Logit on a pure {0,1} observation is ±∞
            # without smoothing; we defer that to the live path where
            # choice_probs is always present for logprob-scored runs.
            if isinstance(choice, str) and choice in mapping:
                response_matrix[k, col_idx_0] = float(mapping[choice])
        return

    if fc_pair_high is not None and item_id in fc_pair_high:
        # fc_pair: soft 2·P(high) − 1 in [−1, +1] when ``choice_probs`` is
        # supplied (logprob-scored path); hard ±1 fallback when only the
        # argmax letter is available. ``choice_probs`` is already softmax-
        # normalized over the found {A, B} letters by
        # parse_top_logprobs_to_choice_probs, so ``probs.get(high, 0.0)``
        # IS P(high) — no further re-normalization needed. When only one
        # of A/B appears in top-k the entry is 1.0 and the soft encoding
        # collapses to ±1 (matches the hard-argmax fallback exactly).
        high = fc_pair_high[item_id]
        if choice_probs:
            p_high = float(choice_probs.get(high, 0.0))
            response_matrix[k, col_idx_0] = 2.0 * p_high - 1.0
            return
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
        # Likert: 1..likert_scale, apply reverse keying.
        # When choice_probs is supplied (logprob-scored path), compute
        # the soft expected value Σ i · P(i) over digit keys — gives a
        # continuous float in [1, likert_scale] instead of the argmax
        # integer. Reverse-keyed items use ``(likert_scale + 1) − score``
        # so the polarity convention matches the hard-scored path.
        #
        # Note on the scale parameter: we must use the configured scale,
        # not ``max(digits present in top-k)``. Strongly-polarised
        # responses commonly have top-k logprobs that do not span the
        # full 1..N scale (e.g. only digits {1, 2} present when the
        # model is strongly disagreeing). Using the observed max would
        # reverse-code a "strongly disagree" response as "slightly
        # disagree" instead of its true complement "strongly agree".
        #
        # Falls through to argmax encoding when the caller didn't
        # provide digit probs.
        if choice_probs and all(
            isinstance(k_, (int, float)) or (isinstance(k_, str) and k_.isdigit())
            for k_ in choice_probs.keys()
        ):
            numeric_probs = {int(k_): float(v) for k_, v in choice_probs.items()}
            if numeric_probs:
                score = sum(i * p for i, p in numeric_probs.items())
                if likert_reverse.get(item_id, False):
                    score = (likert_scale + 1) - score
                response_matrix[k, col_idx_0] = float(score)
                return
        score = int(choice)
        if likert_reverse.get(item_id, False):
            score = (likert_scale + 1) - score
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
    likert_scale: int = 5,
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
        likert_scale=likert_scale,
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
