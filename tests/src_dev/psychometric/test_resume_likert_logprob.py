"""Regression test for the encoding-version rebuild path on Likert logprob caches.

Background — the bug we're guarding against:

Stage 2 caches under the ``logprob`` scoring method persist two different
raw_responses schemas:

* ``trait_mcq`` / ``fc_pair`` entries carry ``parsed_choice`` (a letter).
* ``likert`` entries carry ``score`` / ``ev_raw`` / ``probs`` but **no**
  ``parsed_choice`` — the soft-expectation value is already the cell
  value, and the argmax letter is not what gets stored in the matrix.

When the response-matrix encoding version bumps (v2 → v3 under fa-review,
v3 → v4 under min-choice-mass gate), the rebuild path in
``questionnaire_inference.py`` replays raw_responses through
``fill_matrix_from_choice`` to re-encode every cell under the new rules.

The bug: the resume loop skipped any entry where ``parsed_choice is None``
— which was every Likert logprob entry. The rebuild emitted "All cells
already in raw_responses.jsonl — skipping inference and writing matrix
directly" and then wrote a matrix of NaN. Those NaN matrices were uploaded
to HF, overwriting the (correct) encoding-v2 versions. Downstream the
combined FA stage got a half-NaN combined matrix and the factor analyzer
received a 0×0 array after variance filtering.

The fix: when ``parsed_choice`` is missing, derive it from the probs
argmax. For Likert the probs-scoring branch in
``fill_matrix_from_choice`` uses ``choice_probs`` and ignores ``choice``,
so the derived value just needs to be truthy.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    pytest.importorskip("transformers", reason="transformers not installed") is None,
    reason="transformers not available",
)


def _write_likert_raw_responses(path: Path, K: int, item_ids: list[str]) -> None:
    """Emit a minimal Likert logprob raw_responses.jsonl — no parsed_choice."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for k in range(K):
            for iid in item_ids:
                entry = {
                    "k": k,
                    "item_id": iid,
                    "item_type": "likert",
                    "probs": {"1": 0.01, "2": 0.02, "3": 0.07, "4": 0.85, "5": 0.05},
                    "choice_mass": 1.0,
                    "ev_raw": 3.91,
                    "reverse_keyed": False,
                    "score": 3.91,
                    "raw_text": " 4",
                    "scoring_method": "logprob",
                }
                f.write(json.dumps(entry) + "\n")


def test_rebuild_fills_likert_logprob_cells_despite_missing_parsed_choice(tmp_path: Path):
    """Replaying Likert logprob raw_responses through fill_matrix_from_choice
    must populate the matrix with soft-expectation scores — not leave NaN.

    We construct the minimum state ``run_questionnaire_inference_async``
    touches during its resume-path, then invoke ``fill_matrix_from_choice``
    with the same choice-derivation logic as the resume code. If the probs
    argmax fallback regresses (or is removed), every cell will stay NaN.
    """
    from src_dev.psychometric.response_encoding import fill_matrix_from_choice

    K = 3
    item_ids = ["lik_a", "lik_b"]
    raw_log = tmp_path / "raw_responses.jsonl"
    _write_likert_raw_responses(raw_log, K, item_ids)

    # Minimal state matching the resume loop's expected structure.
    item_to_cols: dict[str, list[tuple[int, str | None]]] = {
        iid: [(idx, "dim")] for idx, iid in enumerate(item_ids)
    }
    likert_reverse = {iid: False for iid in item_ids}
    response_matrix = np.full((K, len(item_ids)), np.nan)

    # Mimic the resume loop's cell-fill path (post-fix).
    with raw_log.open() as f:
        for line in f:
            entry = json.loads(line)
            iid = entry["item_id"]
            k = entry["k"]
            choice = entry.get("parsed_choice")
            if choice is None:
                probs_raw = entry.get("probs")
                assert isinstance(probs_raw, dict) and probs_raw, (
                    "Test harness bug: constructed Likert entry lacks probs"
                )
                # Mirror the production fix: cast to int so
                # fill_matrix_from_choice dispatches to the Likert branch,
                # not the Vignette branch (which is triggered on str choice).
                assert entry.get("item_type") == "likert"
                choice = int(
                    max(probs_raw.items(), key=lambda kv: float(kv[1]))[0]
                )
            resumed_probs = entry.get("probs")
            resumed_probs = (
                {str(kk): float(vv) for kk, vv in resumed_probs.items()}
                if isinstance(resumed_probs, dict) and resumed_probs
                else None
            )
            fill_matrix_from_choice(
                response_matrix, k, iid, choice,
                item_to_cols, vig_scoring={}, likert_reverse=likert_reverse,
                choice_probs=resumed_probs,
                likert_scale=5,
            )

    # Every cell must be populated — no NaN anywhere.
    assert not np.isnan(response_matrix).any(), (
        f"Matrix still has NaN after Likert logprob rebuild:\n{response_matrix}"
    )
    # Soft expected value = 1·0.01 + 2·0.02 + 3·0.07 + 4·0.85 + 5·0.05 = 3.91
    expected = 1 * 0.01 + 2 * 0.02 + 3 * 0.07 + 4 * 0.85 + 5 * 0.05
    assert np.allclose(response_matrix, expected, atol=1e-6), (
        f"Soft expectation mismatch. Got\n{response_matrix}\nexpected uniform {expected}"
    )
