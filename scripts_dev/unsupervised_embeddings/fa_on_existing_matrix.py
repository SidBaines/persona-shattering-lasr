"""Subset example: run Stage 3 (factor analysis) + Stage 5 (validation) on an
already-computed response matrix.

Use when you want to re-run the FA / validation stages against an existing
``questionnaire/`` directory without regenerating rollouts or re-administering
the questionnaire. Typical use cases:

* Re-run FA after tweaking ``FA_N_FACTORS_OVERRIDE`` / ``FA_ROTATIONS`` /
  ``MIN_ITEM_VARIANCE``.
* Re-run validation with a different subset of tests or different thresholds.
* Point at a multi-pair combined directory built by a previous full run.

This script is a **config + stage calls** template — edit the four paths
at the top and the FA/validation knobs below, then run it. Everything heavy
is in ``src_dev.psychometric``; this script is ~100 lines.
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
    RunContext,
    ValidationStageConfig,
    run_stage_factor_analysis,
    run_stage_validation,
)

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
# CONFIG — edit these four paths to point at your cached pair
# ═════════════════════════════════════════════════════════════════════════════

# Questionnaire directory produced by Stage 2 of the main script. For single-
# pair runs this is the per-pair directory; for multi-pair runs point at the
# combined dir (e.g. ``scratch/psychometric_fa/combined-R[A+B]-Q[...]``).
QUESTIONNAIRE_DIR = Path(
    "scratch/psychometric_fa/"
    "questionnaire-rollouts-llama318binstruct-t1.0-10t-1000p-seed432-scenarios_v1"
    "-q_v5-likert-direct"
)

# Rollout directory (or directories, for a combined run) — needed by
# Stage 3 for archetype-aware plots and by Stage 5 for variance-decomp
# scenario/archetype lookups. Single-entry list for a single-pair run;
# one entry per rollout preset for a combined run.
ROLLOUT_DIRS: list[Path] = [
    Path("scratch/psychometric_fa/rollouts-llama318binstruct-t1.0-10t-1000p-seed432-scenarios_v1"),
]

# Questionnaire JSON — used by Stage 3 to detect trait_mcq items for the
# trait-oriented FA sub-pass. Ignored for Likert-only / fc_pair-only
# questionnaires.
QUESTIONNAIRE_PATH = Path(
    "datasets/psychometric_questionnaires/psychometric_questionnaire_v5.json"
)

# Output goes under this root. The stages produce ``factor_analysis/``,
# ``factor_analysis_trait_oriented/``, ``validation/``, and ``labeling/``
# subdirectories inside it.
OUTPUT_DIR = QUESTIONNAIRE_DIR  # write alongside Stage 2 artefacts

# ═════════════════════════════════════════════════════════════════════════════
# Stage knobs — override the defaults on FactorAnalysisStageConfig /
# ValidationStageConfig here. Anything not set uses the dataclass defaults.
# ═════════════════════════════════════════════════════════════════════════════

FA_METHOD = "principal"
FA_N_FACTORS_OVERRIDE: int | None = 7
FA_ROTATIONS = ("oblimin", "varimax")
FA_PER_BLOCK_PASSES = True
MIN_ITEM_VARIANCE = 0.1

# Which Stage-5 tests to run — drop entries to skip the expensive ones.
VALIDATION_TESTS = frozenset({
    "shuffle_control",
    "item_holdout",
    "stability_icc",
    "variance_decomp",
    "trait_convergence",
    # "stability_sweep_random50",
    # "stability_sweep_loao",
    # "stability_sweep_loso_top10",
    "k_sensitivity",
    # "persona_item_cv",
    "n_factors_suggest",
    # "bootstrap_loadings",
    # "split_half_congruence",
    "cv_k_curve",
    # "external_predictivity",
})

# NUM_ROLLOUTS_PER_PROMPT informs the stability_icc skip path. Set to match
# the source rollout preset (1 for B, 2 for A in the canonical presets).
NUM_ROLLOUTS_PER_PROMPT = 2


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # ── Load Stage-2 artefacts ──────────────────────────────────────────
    q_dir = QUESTIONNAIRE_DIR / "questionnaire"
    response_matrix = np.load(q_dir / "response_matrix.npy")
    with open(q_dir / "metadata.jsonl") as f:
        metadata = [json.loads(line) for line in f if line.strip()]
    with open(q_dir / "items.json") as f:
        column_defs = json.load(f)

    print(f"Loaded response matrix: {response_matrix.shape}")
    print(f"  metadata rows: {len(metadata)}")
    print(f"  column defs:   {len(column_defs)}")

    # ── Build a minimal RunContext ──────────────────────────────────────
    # Most fields are informational only for this subset path — Stage 3/5
    # only reach into ``effective_questionnaire_dir`` and ``rollout_dir``.
    ctx = RunContext(
        scratch_root=QUESTIONNAIRE_DIR.parent,
        hf_repo_id="",  # unused; Stages 3 & 5 never touch HF
        rollout_run_id=ROLLOUT_DIRS[0].name,
        questionnaire_run_id=QUESTIONNAIRE_DIR.name,
        rollout_dir=ROLLOUT_DIRS[0],
        questionnaire_dir=OUTPUT_DIR,
        effective_questionnaire_dir=OUTPUT_DIR,
        is_multi_preset=len(ROLLOUT_DIRS) > 1,
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
        high_variance_persona_drop_pct=0.0,
        fa_blocks=tuple(sorted({cd["block"] for cd in column_defs if cd.get("block")})),
        fa_per_block_passes=FA_PER_BLOCK_PASSES,
        fc_pair_sign_alignment=True,
    )
    fa_stage_result = run_stage_factor_analysis(
        fa_cfg,
        response_matrix, metadata, column_defs,
        items=column_defs,
        seed=SEED,
        questionnaire_path=QUESTIONNAIRE_PATH,
        rollout_dirs=ROLLOUT_DIRS,
        labeling_dir=None,
    )
    fa_results = fa_stage_result.results_by_rotation
    print(f"\n[Stage 3] Produced {len(fa_results)} FA variants: {list(fa_results.keys())}")

    # ── Stage 5: validation ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[Stage 5] Validation")
    print("=" * 60)
    val_cfg = ValidationStageConfig(
        ctx=ctx,
        tests_to_run=VALIDATION_TESTS,
    )
    val_stage_result = run_stage_validation(
        val_cfg,
        response_matrix, metadata, column_defs, fa_results,
        seed=SEED,
        num_rollouts_per_prompt=NUM_ROLLOUTS_PER_PROMPT,
        fa_method=FA_METHOD,
        rotations=FA_ROTATIONS,
        min_item_variance=MIN_ITEM_VARIANCE,
        rollout_dirs=ROLLOUT_DIRS,
    )
    print(f"\n[Stage 5] Results written to {val_stage_result.output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
