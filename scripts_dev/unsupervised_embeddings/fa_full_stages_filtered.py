"""Run the full FA pipeline (Stage 3 + Stage 4 labeling + Stage 5 validation)
on a filtered slice of an already-computed combined response matrix.

Based on ``fa_on_existing_matrix.py`` but with two filters applied before
Stage 3:

* **Row filter**: keep only rollouts with ``rollout_preset_key`` in
  ``INCLUDE_ROLLOUT_KEYS`` (e.g. ``["B"]`` to drop the scenarios-v1 /
  glm-4.7-flash preset A rows).
* **Item filter**: keep only columns whose ``questionnaire_version`` is in
  ``INCLUDE_VERSIONS`` (e.g. ``["v5", "trait_ocean_v1"]`` to drop the
  fc_pair v6_fc_draft items).

Filtered artefacts are written into a new sibling directory under the same
scratch root so the original combined dir is untouched. Stage 3/4/5 then
target that new directory (via ``ctx.effective_questionnaire_dir``), so
``factor_analysis/``, ``labeling/`` and ``validation/`` outputs all land
together under the filtered slice.

Justification for the default filters (from prior item-split validation
sweeps on this matrix):

* Preset A (scenarios_v1 / glm user-sim) is the dominant noise source for
  cross-questionnaire replicability; B-only runs replicate ~0.78-0.88 on F0.
* ``v6_fc_draft`` items don't meaningfully affect random-halves replication
  under stratified sampling and complicate the cross-version comparison.
* ``MIN_ITEM_VARIANCE=0.1`` (per-block relative) gave the best replication;
  aggressive top-N variance selection lost secondary-factor signal.
"""

from __future__ import annotations

# ── Seeds (before any stochastic imports) ───────────────────────────────────
import random

import numpy as np

SEED = 436
random.seed(SEED)
np.random.seed(SEED)

import json
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src_dev.psychometric import (
    FactorAnalysisStageConfig,
    LabelingStageConfig,
    RunContext,
    ValidationStageConfig,
    run_stage_factor_analysis,
    run_stage_labeling,
    run_stage_validation,
)

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
# PATHS — point at the combined Stage-2 dir and the two source rollout dirs
# ═════════════════════════════════════════════════════════════════════════════

SOURCE_QUESTIONNAIRE_DIR = Path(
    "scratch/psychometric_fa/"
    "combined-R[A+B]-Q[v5+v6_fc_draft+trait_ocean_v1]"
)

# All rollout dirs referenced by the source combined matrix. We pass the
# subset matching ``INCLUDE_ROLLOUT_KEYS`` to the stages below.
ALL_ROLLOUT_DIRS_BY_KEY: dict[str, Path] = {
    "A": Path(
        "scratch/psychometric_fa/"
        "rollouts-llama318binstruct-t1.0-10t-1000p-seed432-scenarios_v1"
    ),
    "B": Path(
        "scratch/psychometric_fa/"
        "rollouts-llama318binstruct-t1.0-15t-2500p-seed436-scenarios_v2-uprompt_v6"
    ),
}

# Questionnaire JSONs. Only the versions in ``INCLUDE_VERSIONS`` are strictly
# needed downstream, but the trait-oriented FA sub-pass (Stage 3b) needs
# ``trait_ocean_v1`` specifically.
QUESTIONNAIRE_PATH_TRAIT_OCEAN = Path(
    "datasets/psychometric_questionnaires/psychometric_questionnaire_trait_ocean_v1.json"
)

# ═════════════════════════════════════════════════════════════════════════════
# FILTERS
# ═════════════════════════════════════════════════════════════════════════════

INCLUDE_ROLLOUT_KEYS: list[str] = ["B"]
INCLUDE_VERSIONS: list[str] = ["v5", "trait_ocean_v1"]

# Per-preset rollouts-per-prompt (for stability_icc skip path in validation).
# Preset A = 2 rollouts/prompt, preset B = 1. For a mixed run we take the min.
NUM_ROLLOUTS_PER_PROMPT = min(
    2 if "A" in INCLUDE_ROLLOUT_KEYS else 10**6,
    1 if "B" in INCLUDE_ROLLOUT_KEYS else 10**6,
)

# ═════════════════════════════════════════════════════════════════════════════
# STAGE 3 knobs
# ═════════════════════════════════════════════════════════════════════════════

FA_METHOD = "principal"
FA_ROTATIONS: tuple[str, ...] = ("oblimin", "varimax")
FA_N_FACTORS_OVERRIDE: int | None = 4
FA_PER_BLOCK_PASSES = True
MIN_ITEM_VARIANCE = 0.1
HIGH_VARIANCE_PERSONA_DROP_PCT = 0.0

# ═════════════════════════════════════════════════════════════════════════════
# STAGE 4 knobs (labeling)
# ═════════════════════════════════════════════════════════════════════════════

LABELLER_MODE: str = "manual"  # "auto" | "manual"
LABELLER_MODEL = "openai/gpt-5.4-nano"
LABELLER_PROVIDER = "openrouter"
TOP_LOADING_ITEMS = 10
LABELLER_MAX_NEW_TOKENS = 500_000
LABELLER_EMPTY_RESPONSE_RETRIES = 2
LABELLER_REASONING: dict | None = {"effort": "high"}
LABELLER_USE_CLAUDE_CLI = False
LABELLER_CLAUDE_CLI_PATH = "claude"
LABELLER_CLAUDE_CLI_MODEL = "opus"
LABELLER_CLAUDE_CLI_TIMEOUT = 3600
LABELLER_CLAUDE_CLI_EFFORT = "high"

# ═════════════════════════════════════════════════════════════════════════════
# STAGE 5 knobs (validation)
# ═════════════════════════════════════════════════════════════════════════════

VALIDATION_TESTS = frozenset({
    "shuffle_control",
    "item_holdout",
    "stability_icc",
    "variance_decomp",
    "trait_convergence",
    "stability_sweep_random50",
    "stability_sweep_loao",
    "stability_sweep_loso_top10",
    "k_sensitivity",
    "persona_item_cv",
    "n_factors_suggest",
    "bootstrap_loadings",
    "split_half_congruence",
    "cv_k_curve",
    "external_predictivity",
})

STABILITY_N_PROMPTS = 100
HOLDOUT_N_ITEMS = 20
HOLDOUT_R2_FLOOR = 0.05
STABILITY_SWEEP_N_RANDOM_SPLITS = 10
STABILITY_SWEEP_LOSO_TOP_N = 10
STABILITY_SWEEP_PA_ITERATIONS = 50
STABILITY_SWEEP_PASS_THRESHOLD_PHI = 0.80
VARIANCE_DECOMP_FIELDS = ("archetype", "scenario_id", "input_group_id")
VARIANCE_DECOMP_SCENARIO_CEILING = 0.30
VARIANCE_DECOMP_ARCHETYPE_FLOOR = 0.05
TRAIT_CONVERGENCE_HIT_THRESHOLD = 0.30
TRAIT_CONVERGENCE_MIN_HITS = 3
TRAIT_CONVERGENCE_N_BOOTSTRAP = 1000
PERSONA_ITEM_CV_SPLIT = 0.7
PERSONA_ITEM_CV_N_TRIALS = 5
PERSONA_ITEM_CV_SUBSET_STRATEGY = "random"
PERSONA_ITEM_CV_N_OUTER_SPLITS = 20
PERSONA_ITEM_CV_BOOTSTRAP_CI = 95.0
K_SENSITIVITY_MATCH_THRESHOLD = 0.85
K_SENSITIVITY_INDEPENDENT_THRESHOLD = 0.60
N_FACTORS_SUGGEST_METHODS = (
    "parallel", "map", "ekc", "acceleration", "kaiser",
    "cv_reconstruction", "optimal_coordinates", "scree_elbow",
)
N_FACTORS_SUGGEST_K_MAX = 15
N_FACTORS_SUGGEST_CV_N_FOLDS = 5
N_FACTORS_SUGGEST_PA_ITERATIONS = 100
BOOTSTRAP_LOADINGS_N_BOOT = 500
BOOTSTRAP_LOADINGS_CONFIDENCE = 95.0
SPLIT_HALF_CONGRUENCE_N_ITERS = 100
SPLIT_HALF_CONGRUENCE_PASS_THRESHOLD_PHI = 0.85
CV_K_CURVE_K_MAX = 15
CV_K_CURVE_N_FOLDS = 5
EXTERNAL_PREDICTIVITY_N_FOLDS = 5
EXTERNAL_PREDICTIVITY_RIDGE_ALPHA = 1.0
EXTERNAL_PREDICTIVITY_PASS_R2 = 0.05
EXTERNAL_PREDICTIVITY_BOOTSTRAP_CI = 95.0

# ═════════════════════════════════════════════════════════════════════════════
# OUTPUT DIR — derived from filters so parallel runs don't collide
# ═════════════════════════════════════════════════════════════════════════════

_r_tag = "+".join(sorted(INCLUDE_ROLLOUT_KEYS))
_v_tag = "+".join(sorted(INCLUDE_VERSIONS))
OUTPUT_DIR = SOURCE_QUESTIONNAIRE_DIR.parent / (
    f"filtered-R[{_r_tag}]-Q[{_v_tag}]-minvar{MIN_ITEM_VARIANCE:g}-k{FA_N_FACTORS_OVERRIDE}"
)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════


def _filter_source(
    response_matrix: np.ndarray,
    metadata: list[dict],
    column_defs: list[dict],
) -> tuple[np.ndarray, list[dict], list[dict]]:
    """Apply row filter (rollout keys) + column filter (questionnaire versions)."""
    row_mask = np.array([
        str(m.get("rollout_preset_key", "")) in set(INCLUDE_ROLLOUT_KEYS)
        for m in metadata
    ])
    col_mask = np.array([
        str(c.get("questionnaire_version", "")) in set(INCLUDE_VERSIONS)
        for c in column_defs
    ])
    rm = response_matrix[row_mask][:, col_mask]
    meta_f = [m for m, k in zip(metadata, row_mask) if k]
    cols_f = [c for c, k in zip(column_defs, col_mask) if k]

    from collections import Counter
    print(f"  row filter: kept {rm.shape[0]}/{response_matrix.shape[0]} rows "
          f"(rollout_preset_key in {INCLUDE_ROLLOUT_KEYS})")
    print(f"  col filter: kept {rm.shape[1]}/{response_matrix.shape[1]} cols "
          f"(questionnaire_version in {INCLUDE_VERSIONS})")
    print(f"    by version: {dict(Counter(c['questionnaire_version'] for c in cols_f))}")
    print(f"    by block:   {dict(Counter(c['block'] for c in cols_f))}")
    return rm, meta_f, cols_f


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # ── Load source artefacts ───────────────────────────────────────────
    src_q_dir = SOURCE_QUESTIONNAIRE_DIR / "questionnaire"
    response_matrix = np.load(src_q_dir / "response_matrix.npy")
    with open(src_q_dir / "metadata.jsonl") as f:
        metadata = [json.loads(line) for line in f if line.strip()]
    with open(src_q_dir / "items.json") as f:
        column_defs = json.load(f)
    print(f"Source matrix: {response_matrix.shape}, metadata={len(metadata)}, items={len(column_defs)}")

    # ── Apply filters ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Filtering to rollouts={INCLUDE_ROLLOUT_KEYS}, versions={INCLUDE_VERSIONS}")
    print("=" * 60)
    response_matrix, metadata, column_defs = _filter_source(
        response_matrix, metadata, column_defs
    )

    # ── Write filtered artefacts into OUTPUT_DIR for provenance ─────────
    out_q_dir = OUTPUT_DIR / "questionnaire"
    out_q_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_q_dir / "response_matrix.npy", response_matrix)
    with open(out_q_dir / "metadata.jsonl", "w") as f:
        for m in metadata:
            f.write(json.dumps(m) + "\n")
    with open(out_q_dir / "items.json", "w") as f:
        json.dump(column_defs, f, indent=2)
    with open(OUTPUT_DIR / "filter_config.json", "w") as f:
        json.dump({
            "source": str(SOURCE_QUESTIONNAIRE_DIR),
            "include_rollout_keys": INCLUDE_ROLLOUT_KEYS,
            "include_versions": INCLUDE_VERSIONS,
            "num_rollouts_per_prompt": NUM_ROLLOUTS_PER_PROMPT,
            "fa_method": FA_METHOD,
            "fa_rotations": list(FA_ROTATIONS),
            "fa_n_factors_override": FA_N_FACTORS_OVERRIDE,
            "min_item_variance": MIN_ITEM_VARIANCE,
            "seed": SEED,
        }, f, indent=2)
    print(f"  wrote filtered artefacts to {out_q_dir}")

    # ── Build RunContext pointing at OUTPUT_DIR ─────────────────────────
    rollout_dirs = [ALL_ROLLOUT_DIRS_BY_KEY[k] for k in INCLUDE_ROLLOUT_KEYS]
    ctx = RunContext(
        scratch_root=SOURCE_QUESTIONNAIRE_DIR.parent,
        hf_repo_id="",
        rollout_run_id=rollout_dirs[0].name,
        questionnaire_run_id=OUTPUT_DIR.name,
        rollout_dir=rollout_dirs[0],
        questionnaire_dir=OUTPUT_DIR,
        effective_questionnaire_dir=OUTPUT_DIR,
        is_multi_preset=len(rollout_dirs) > 1,
        provenance={"entry_point": __file__},
    )

    # ── Stage 3: factor analysis ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[Stage 3] Factor analysis")
    print("=" * 60)
    fa_cfg = FactorAnalysisStageConfig(
        ctx=ctx,
        method=FA_METHOD,
        n_factors_override=FA_N_FACTORS_OVERRIDE,
        rotations=FA_ROTATIONS,
        residualize_options=(False,),
        min_item_variance=MIN_ITEM_VARIANCE,
        high_variance_persona_drop_pct=HIGH_VARIANCE_PERSONA_DROP_PCT,
        fa_blocks=tuple(sorted({cd["block"] for cd in column_defs if cd.get("block")})),
        fa_per_block_passes=FA_PER_BLOCK_PASSES,
        fc_pair_sign_alignment=True,
    )
    fa_stage_result = run_stage_factor_analysis(
        fa_cfg,
        response_matrix, metadata, column_defs,
        items=column_defs,
        seed=SEED,
        questionnaire_path=QUESTIONNAIRE_PATH_TRAIT_OCEAN,
        rollout_dirs=rollout_dirs,
        labeling_dir=None,
    )
    fa_results = fa_stage_result.results_by_rotation
    print(f"\n[Stage 3] Produced {len(fa_results)} FA variants: {list(fa_results.keys())}")

    # ── Stage 4: labeling ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[Stage 4] Factor labeling")
    print("=" * 60)
    lbl_cfg = LabelingStageConfig(
        ctx=ctx,
        mode=LABELLER_MODE,
        model=LABELLER_MODEL,
        provider=LABELLER_PROVIDER,
        top_loading_items=TOP_LOADING_ITEMS,
        max_new_tokens=LABELLER_MAX_NEW_TOKENS,
        empty_response_retries=LABELLER_EMPTY_RESPONSE_RETRIES,
        reasoning=LABELLER_REASONING,
        use_claude_cli=LABELLER_USE_CLAUDE_CLI,
        claude_cli_path=LABELLER_CLAUDE_CLI_PATH,
        claude_cli_model=LABELLER_CLAUDE_CLI_MODEL,
        claude_cli_timeout=LABELLER_CLAUDE_CLI_TIMEOUT,
        claude_cli_effort=LABELLER_CLAUDE_CLI_EFFORT,
    )
    run_stage_labeling(
        lbl_cfg, fa_results, column_defs,
        rollout_dirs=rollout_dirs,
    )

    # ── Stage 5: validation ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[Stage 5] Validation")
    print("=" * 60)
    val_cfg = ValidationStageConfig(
        ctx=ctx,
        tests_to_run=VALIDATION_TESTS,
        stability_n_prompts=STABILITY_N_PROMPTS,
        holdout_n_items=HOLDOUT_N_ITEMS,
        holdout_r2_floor=HOLDOUT_R2_FLOOR,
        stability_sweep_n_random_splits=STABILITY_SWEEP_N_RANDOM_SPLITS,
        stability_sweep_loso_top_n=STABILITY_SWEEP_LOSO_TOP_N,
        stability_sweep_pa_iterations=STABILITY_SWEEP_PA_ITERATIONS,
        stability_sweep_pass_threshold_phi=STABILITY_SWEEP_PASS_THRESHOLD_PHI,
        variance_decomp_fields=VARIANCE_DECOMP_FIELDS,
        variance_decomp_scenario_ceiling=VARIANCE_DECOMP_SCENARIO_CEILING,
        variance_decomp_archetype_floor=VARIANCE_DECOMP_ARCHETYPE_FLOOR,
        trait_convergence_hit_threshold=TRAIT_CONVERGENCE_HIT_THRESHOLD,
        trait_convergence_min_hits=TRAIT_CONVERGENCE_MIN_HITS,
        trait_convergence_n_bootstrap=TRAIT_CONVERGENCE_N_BOOTSTRAP,
        persona_item_cv_split=PERSONA_ITEM_CV_SPLIT,
        persona_item_cv_n_trials=PERSONA_ITEM_CV_N_TRIALS,
        persona_item_cv_subset_strategy=PERSONA_ITEM_CV_SUBSET_STRATEGY,
        persona_item_cv_n_outer_splits=PERSONA_ITEM_CV_N_OUTER_SPLITS,
        persona_item_cv_bootstrap_ci=PERSONA_ITEM_CV_BOOTSTRAP_CI,
        k_sensitivity_match_threshold=K_SENSITIVITY_MATCH_THRESHOLD,
        k_sensitivity_independent_threshold=K_SENSITIVITY_INDEPENDENT_THRESHOLD,
        n_factors_suggest_methods=N_FACTORS_SUGGEST_METHODS,
        n_factors_suggest_k_max=N_FACTORS_SUGGEST_K_MAX,
        n_factors_suggest_cv_n_folds=N_FACTORS_SUGGEST_CV_N_FOLDS,
        n_factors_suggest_pa_iterations=N_FACTORS_SUGGEST_PA_ITERATIONS,
        bootstrap_loadings_n_boot=BOOTSTRAP_LOADINGS_N_BOOT,
        bootstrap_loadings_confidence=BOOTSTRAP_LOADINGS_CONFIDENCE,
        split_half_congruence_n_iters=SPLIT_HALF_CONGRUENCE_N_ITERS,
        split_half_congruence_pass_threshold_phi=SPLIT_HALF_CONGRUENCE_PASS_THRESHOLD_PHI,
        cv_k_curve_k_max=CV_K_CURVE_K_MAX,
        cv_k_curve_n_folds=CV_K_CURVE_N_FOLDS,
        external_predictivity_n_folds=EXTERNAL_PREDICTIVITY_N_FOLDS,
        external_predictivity_ridge_alpha=EXTERNAL_PREDICTIVITY_RIDGE_ALPHA,
        external_predictivity_pass_r2=EXTERNAL_PREDICTIVITY_PASS_R2,
        external_predictivity_bootstrap_ci=EXTERNAL_PREDICTIVITY_BOOTSTRAP_CI,
    )
    val_stage_result = run_stage_validation(
        val_cfg,
        response_matrix, metadata, column_defs, fa_results,
        seed=SEED,
        num_rollouts_per_prompt=NUM_ROLLOUTS_PER_PROMPT,
        fa_method=FA_METHOD,
        rotations=FA_ROTATIONS,
        min_item_variance=MIN_ITEM_VARIANCE,
        high_variance_persona_drop_pct=HIGH_VARIANCE_PERSONA_DROP_PCT,
        rollout_dirs=rollout_dirs,
    )
    print(f"\n[Stage 5] Results written to {val_stage_result.output_dir}")
    print(f"\nAll stages done. Output root: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
