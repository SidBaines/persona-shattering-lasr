"""Regression tests for ``_rollout_run_id`` in the psychometric FA script.

``psychometric_rollout_fa.py`` builds a deterministic run-id from each
preset; Stage-1 / Stage-2 HF caches on
``persona-shattering-lasr/psychometric-fa-runs`` are keyed by this
string. Changing any id invalidates the corresponding cache and forces
regeneration / re-administration, which for preset B would mean
re-running 2500 personas × 100 items on Llama-3.1-8B — so we treat the
current ids for A/B as a frozen contract and pin them here.

External presets are pinned too, ensuring future refactors don't
silently collide with existing caches or with each other.
"""

from __future__ import annotations

import pytest

from scripts_dev.unsupervised_embeddings import psychometric_rollout_fa as m


# Frozen expected strings. Only update these in a commit that also bumps
# the on-HF cache (i.e. regenerates / re-administers under the new id).
FROZEN_RUN_IDS: dict[str, str] = {
    # Generation presets — must never change without an HF cache
    # migration. Breaking A means losing the v1 1000-prompt cache;
    # breaking B means losing the v2 2500-prompt cache.
    "A": "rollouts-llama318binstruct-t1.0-10t-1000p-seed432-scenarios_v1",
    "B": "rollouts-llama318binstruct-t1.0-15t-2500p-seed436-scenarios_v2-uprompt_v6",
    # External presets — pinned so M2+ refactors don't silently collide
    # with each other or with a future generation preset.
    "kwai_swe": (
        "rollouts-external-kwai_swe_smith-qwen38b-500p-seed436"
    ),
    "swe_rebench": (
        "rollouts-external-swe_rebench-qwen3coder480ba35binstruct-"
        "500p-seed436-f_resolved"
    ),
    # PRISM per-model presets.
    "prism_mistral_7b_v01": (
        "rollouts-external-prism_open-mistral7binstructv01-500p-seed436-"
        "f_mistral7bv01"
    ),
    "prism_mistral_7b_v01_n50": (
        "rollouts-external-prism_open-mistral7binstructv01-50p-seed436-"
        "f_mistral7bv01"
    ),
    "prism_zephyr_7b_beta": (
        "rollouts-external-prism_open-zephyr7bbeta-500p-seed436-"
        "f_zephyr7bbeta"
    ),
    "prism_zephyr_7b_beta_n50": (
        "rollouts-external-prism_open-zephyr7bbeta-50p-seed436-"
        "f_zephyr7bbeta"
    ),
    "prism_llama2_7b_chat": (
        "rollouts-external-prism_open-llama27bchathf-500p-seed436-"
        "f_llama2_7b_chat"
    ),
    "prism_llama2_7b_chat_n50": (
        "rollouts-external-prism_open-llama27bchathf-50p-seed436-"
        "f_llama2_7b_chat"
    ),
    "prism_llama2_13b_chat": (
        "rollouts-external-prism_open-llama213bchathf-500p-seed436-"
        "f_llama2_13b_chat"
    ),
    "prism_llama2_13b_chat_n50": (
        "rollouts-external-prism_open-llama213bchathf-50p-seed436-"
        "f_llama2_13b_chat"
    ),
    "prism_falcon_7b_instruct": (
        "rollouts-external-prism_open-falcon7binstruct-500p-seed436-"
        "f_falcon7b_instruct"
    ),
    "prism_oasst_pythia_12b": (
        "rollouts-external-prism_open-oasstsft4pythia12bepoch35-500p-seed436-"
        "f_oasst_pythia_12b"
    ),
    # LMSYS per-model presets (verified via 100k-row scan — all models
    # below have non-trivial counts; Mistral-7B-Instruct is confirmed
    # NOT in LMSYS at any scan depth).
    "lmsys_vicuna_13b_v15_t5": (
        "rollouts-external-lmsys_open-vicuna13bv15-500p-seed436-"
        "f_vicuna13bv15_t5_en"
    ),
    "lmsys_llama2_13b_chat_t5": (
        "rollouts-external-lmsys_open-llama213bchathf-500p-seed436-"
        "f_llama2_13b_chat_t5_en"
    ),
    "lmsys_vicuna_33b_t5": (
        "rollouts-external-lmsys_open-vicuna33bv13-500p-seed436-"
        "f_vicuna33b_t5_en"
    ),
    "lmsys_wizardlm_13b_t5": (
        "rollouts-external-lmsys_open-wizardlm13bv12-500p-seed436-"
        "f_wizardlm13b_t5_en"
    ),
    "lmsys_llama2_7b_chat_t5": (
        "rollouts-external-lmsys_open-llama27bchathf-500p-seed436-"
        "f_llama2_7b_chat_t5_en"
    ),
    "lmsys_koala_13b_t5": (
        "rollouts-external-lmsys_open-koala13bhf-500p-seed436-"
        "f_koala13b_t5_en"
    ),
    "lmsys_mpt_7b_chat_t5": (
        "rollouts-external-lmsys_open-mpt7bchat-500p-seed436-"
        "f_mpt7b_chat_t5_en"
    ),
}


@pytest.mark.parametrize(
    "preset_key,expected",
    list(FROZEN_RUN_IDS.items()),
    ids=list(FROZEN_RUN_IDS.keys()),
)
def test_rollout_run_id_frozen(preset_key: str, expected: str) -> None:
    """Each preset's run-id matches its frozen fixture."""
    assert m._rollout_run_id(preset_key) == expected


def test_external_and_generation_presets_have_disjoint_keys() -> None:
    """The two preset dicts must never share a key — the orchestrator's
    ``_rollout_preset`` dispatcher depends on this for correct lookup."""
    overlap = set(m.ROLLOUT_PRESETS) & set(m.EXTERNAL_ROLLOUT_PRESETS)
    assert not overlap, f"preset keys overlap: {sorted(overlap)}"


def test_external_run_ids_are_externally_tagged() -> None:
    """Every external-preset run-id must contain the ``-external-``
    segment so it can never collide with a generation-preset cache."""
    for key in m.EXTERNAL_ROLLOUT_PRESETS:
        run_id = m._rollout_run_id(key)
        assert "-external-" in run_id, (
            f"external preset {key!r} run-id missing '-external-' tag: {run_id!r}"
        )


def test_all_external_run_ids_are_unique() -> None:
    """No two external presets may produce the same run-id — otherwise
    they'd share an HF cache path."""
    ids = [m._rollout_run_id(k) for k in m.EXTERNAL_ROLLOUT_PRESETS]
    dupes = {i for i in ids if ids.count(i) > 1}
    assert not dupes, f"duplicate external run-ids: {sorted(dupes)}"
