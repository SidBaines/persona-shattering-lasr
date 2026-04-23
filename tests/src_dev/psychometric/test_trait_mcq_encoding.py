"""Tests for the trait_mcq cell encodings: ``soft_ev`` vs ``logit``.

soft_ev is the historical default: each cell is the trait-aligned soft
expectation Σ P(letter)·answer_mapping[letter] in [0, 1]. logit is a
research alternative that computes the log-odds log(P(high)/(1-P(high)))
with clipping at ``trait_mcq_logit_epsilon``. See the docstring in
``src_dev/psychometric/response_encoding.py: fill_matrix_from_choice``
for the research motivation.

Tests here guard against:

1. soft_ev default behaviour — no regression from the pre-logit math
   (Σ P·mapping with 0/1 answer_mapping reduces to P(high)).
2. logit encoding — computes log(p/(1-p)) on the renormalised-to-letter
   probabilities, with correct sign for high/low pole skew and proper
   clipping on confidently one-sided cells.
3. Hard-argmax fallback — both encodings emit the 0/1 mapping value for
   cells without choice_probs (no false ±∞ without smoothing).
"""

from __future__ import annotations

import numpy as np
import pytest

from src_dev.psychometric.response_encoding import fill_matrix_from_choice


def _single_item_setup(mapping: dict[str, int]):
    """Minimal state for filling a single trait_mcq cell at (k=0, col=0)."""
    item_id = "lik_or_trait_test"
    item_to_cols = {item_id: [(0, "openness")]}  # dim_0 set → goes to trait_mcq/Likert branch
    trait_mcq_mapping = {item_id: mapping}
    matrix = np.full((1, 1), np.nan)
    return matrix, item_id, item_to_cols, trait_mcq_mapping


def test_soft_ev_default_matches_weighted_sum():
    """soft_ev: cell = Σ P(letter) · mapping[letter]. Canonical behaviour."""
    matrix, item_id, item_to_cols, trait_mcq_mapping = _single_item_setup(
        {"A": 0, "B": 1, "C": 0, "D": 1}
    )
    probs = {"A": 0.1, "B": 0.4, "C": 0.1, "D": 0.4}  # P(high) = 0.8
    fill_matrix_from_choice(
        matrix, 0, item_id, "B",
        item_to_cols, vig_scoring={}, likert_reverse={},
        trait_mcq_mapping=trait_mcq_mapping,
        choice_probs=probs,
    )
    assert matrix[0, 0] == pytest.approx(0.8, abs=1e-9)


def test_logit_on_balanced_probs_is_zero():
    """logit: P(high) = 0.5 → log(0.5/0.5) = 0."""
    matrix, item_id, item_to_cols, trait_mcq_mapping = _single_item_setup(
        {"A": 0, "B": 1, "C": 0, "D": 1}
    )
    probs = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}  # P(high) = 0.5
    fill_matrix_from_choice(
        matrix, 0, item_id, "A",
        item_to_cols, vig_scoring={}, likert_reverse={},
        trait_mcq_mapping=trait_mcq_mapping,
        choice_probs=probs,
        trait_mcq_encoding="logit",
    )
    assert matrix[0, 0] == pytest.approx(0.0, abs=1e-9)


def test_logit_matches_analytic_formula():
    """logit: P(high) = 0.8 → log(0.8/0.2) = log(4) ≈ 1.3863."""
    matrix, item_id, item_to_cols, trait_mcq_mapping = _single_item_setup(
        {"A": 0, "B": 1, "C": 0, "D": 1}
    )
    probs = {"A": 0.1, "B": 0.4, "C": 0.1, "D": 0.4}
    fill_matrix_from_choice(
        matrix, 0, item_id, "B",
        item_to_cols, vig_scoring={}, likert_reverse={},
        trait_mcq_mapping=trait_mcq_mapping,
        choice_probs=probs,
        trait_mcq_encoding="logit",
    )
    assert matrix[0, 0] == pytest.approx(np.log(0.8 / 0.2), abs=1e-9)


def test_logit_clipping_prevents_infinity():
    """logit: P(high) ≈ 1.0 should clip to log((1-ε)/ε), not +∞."""
    matrix, item_id, item_to_cols, trait_mcq_mapping = _single_item_setup(
        {"A": 1, "B": 1, "C": 0, "D": 0}
    )
    probs = {"A": 0.5, "B": 0.5, "C": 0.0, "D": 0.0}  # P(high) = 1.0
    eps = 0.005
    fill_matrix_from_choice(
        matrix, 0, item_id, "A",
        item_to_cols, vig_scoring={}, likert_reverse={},
        trait_mcq_mapping=trait_mcq_mapping,
        choice_probs=probs,
        trait_mcq_encoding="logit",
        trait_mcq_logit_epsilon=eps,
    )
    expected = np.log((1.0 - eps) / eps)
    assert np.isfinite(matrix[0, 0])
    assert matrix[0, 0] == pytest.approx(expected, abs=1e-9)


def test_logit_clipping_symmetric_on_low_pole():
    """logit: P(high) ≈ 0.0 should clip to log(ε/(1-ε)) — negative of the
    high-pole clipped value."""
    matrix, item_id, item_to_cols, trait_mcq_mapping = _single_item_setup(
        {"A": 0, "B": 0, "C": 1, "D": 1}
    )
    probs = {"A": 0.5, "B": 0.5, "C": 0.0, "D": 0.0}  # P(high) = 0.0
    eps = 0.005
    fill_matrix_from_choice(
        matrix, 0, item_id, "A",
        item_to_cols, vig_scoring={}, likert_reverse={},
        trait_mcq_mapping=trait_mcq_mapping,
        choice_probs=probs,
        trait_mcq_encoding="logit",
        trait_mcq_logit_epsilon=eps,
    )
    expected = np.log(eps / (1.0 - eps))
    assert np.isfinite(matrix[0, 0])
    assert matrix[0, 0] == pytest.approx(expected, abs=1e-9)


def test_hard_argmax_path_identical_for_both_encodings():
    """Both encodings must emit mapping[choice] for hard-argmax cells
    (no choice_probs). Logit without smoothing would give ±∞; the code
    explicitly falls back to the 0/1 mapping value to avoid that."""
    for enc in ("soft_ev", "logit"):
        matrix, item_id, item_to_cols, trait_mcq_mapping = _single_item_setup(
            {"A": 0, "B": 1, "C": 0, "D": 1}
        )
        fill_matrix_from_choice(
            matrix, 0, item_id, "B",  # high-pole
            item_to_cols, vig_scoring={}, likert_reverse={},
            trait_mcq_mapping=trait_mcq_mapping,
            choice_probs=None,  # hard-argmax path
            trait_mcq_encoding=enc,
        )
        assert matrix[0, 0] == 1.0, f"encoding={enc} expected 1.0, got {matrix[0, 0]}"
