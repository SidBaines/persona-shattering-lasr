#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# OCEAN vanton4 training runner — E− (extraversion suppressor) only.
#
# vanton4 promotes the vanton3 per-facet system-prompt behavior to the pipeline
# DEFAULT, with two structural changes vs vanton3:
#   1. Each curated teacher prompt uses ONLY its originating facet's `trait`
#      (no more concatenation of all 12 facet traits into one shared prompt).
#   2. Each LIMA/factual teacher prompt picks one of the 12 facet `trait`
#      strings at RANDOM (seeded via `--seed`, default 123456 → reproducible),
#      replacing the vanton3 `is_lima_fallback=true` addendum entry.
# The legacy "concat all traits into one shared system prompt" behavior is
# still available as an opt-in via `--concat-all-traits-system-prompt` (not
# used here — vanton4 wants the new default).
#
# vanton4's E− constitution content is a byte-for-byte revert to vanton1's
# clean 12-facet content (no first-person addendum). This isolates the
# pipeline change as the sole experimental lever: any delta vs vanton1 E−
# attributes to per-facet scoping + LIMA random-facet injection.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

MODEL="llama-3.1-8b-it"
TEACHER="z-ai/glm-4.5-air"
VERSION="anton4"

# ── (trait, direction, monorepo_trait, monorepo_direction, eval_suffix) ──
RUNS=(
  "extraversion   suppressing  extraversion       suppressor  e_minus"
)

for run in "${RUNS[@]}"; do
  read -r TRAIT DIR MONO_TRAIT MONO_DIR EVAL_NAME <<< "$run"

  CONSTITUTION="${TRAIT}_${DIR}_full_vanton4"
  CONSTITUTION_JSON="scripts_dev/oct_pipeline/ocean/${CONSTITUTION}.json"
  SLIM_JSON="scripts_dev/oct_pipeline/ocean/${CONSTITUTION}_slim.json"
  OUT_DIR="scratch/oct_${TRAIT}_${DIR}_anton4"

  echo ""
  echo "================================================================"
  echo "  ${TRAIT} ${DIR} (${EVAL_NAME}_vanton4)"
  echo "================================================================"
  echo ""

  # ── Train ──
  uv run python scripts_dev/oct_pipeline/run_oct_pipeline.py \
    --model "$MODEL" \
    --teacher-model "$TEACHER" \
    --custom-constitution "$CONSTITUTION_JSON" \
    --introspection-constitution "$SLIM_JSON" \
    --out-dir "$OUT_DIR" \
    --monorepo-category ocean \
    --monorepo-trait "$MONO_TRAIT" \
    --monorepo-direction "$MONO_DIR" \
    --monorepo-version "$VERSION"

  # ── Clean up distilled model to free disk ──
  rm -rf "${OUT_DIR}/models/distilled/"

  # ── Eval: trait ──
  uv run python -m src_dev.evals suite \
    --config-module "scripts_dev.personality_evals.configs.ocean.trait.${EVAL_NAME}_vanton4"

  # ── Eval: mmlu ──
  uv run python -m src_dev.evals suite \
    --config-module "scripts_dev.personality_evals.configs.ocean.mmlu.${EVAL_NAME}_vanton4"

  echo ""
  echo "  ✓ ${TRAIT} ${DIR} complete"
  echo ""
done

echo ""
echo "All runs complete. Stopping RunPod instance..."
runpodctl stop pod "$RUNPOD_POD_ID"
