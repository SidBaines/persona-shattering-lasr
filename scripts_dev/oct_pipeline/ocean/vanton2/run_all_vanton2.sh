#!/usr/bin/env bash
# Rerun A+ (agreeableness amplifying) with a different seed to measure variance.
# Same constitution as vanton1, different seed → different training data & weights.
set -euo pipefail

MODEL="llama-3.1-8b-it"
TEACHER="z-ai/glm-4.5-air"
VERSION="anton2"
SEED=789012

# ── (trait, direction, monorepo_trait, monorepo_direction, eval_suffix) ──
RUNS=(
  "agreeableness  amplifying   agreeableness      amplifier   a_plus"
)

for run in "${RUNS[@]}"; do
  read -r TRAIT DIR MONO_TRAIT MONO_DIR EVAL_NAME <<< "$run"

  CONSTITUTION="${TRAIT}_${DIR}_full_vanton2"
  CONSTITUTION_JSON="scripts_dev/oct_pipeline/ocean/vanton2/${CONSTITUTION}.json"
  SLIM_JSON="scripts_dev/oct_pipeline/ocean/vanton2/${CONSTITUTION}_slim.json"
  OUT_DIR="scratch/oct_${TRAIT}_${DIR}_anton2"

  echo ""
  echo "================================================================"
  echo "  ${TRAIT} ${DIR} (${EVAL_NAME}_vanton2)"
  echo "================================================================"
  echo ""

  # ── Train ──
  uv run --with-requirements scripts_dev/oct_pipeline/uv-oct-requirements.txt python scripts_dev/oct_pipeline/run_oct_pipeline.py \
    --model "$MODEL" \
    --teacher-model "$TEACHER" \
    --custom-constitution "$CONSTITUTION_JSON" \
    --introspection-constitution "$SLIM_JSON" \
    --out-dir "$OUT_DIR" \
    --seed "$SEED" \
    --monorepo-category ocean \
    --monorepo-trait "$MONO_TRAIT" \
    --monorepo-direction "$MONO_DIR" \
    --monorepo-version "$VERSION"

  # ── Clean up distilled model to free disk ──
  rm -rf "${OUT_DIR}/models/distilled/"

  # ── Eval: trait ──
  uv run python -m src_dev.evals suite \
    --config-module "scripts_dev.personality_evals.configs.ocean.trait.vanton2.${EVAL_NAME}_vanton2"

  # ── Eval: mmlu ──
  uv run python -m src_dev.evals suite \
    --config-module "scripts_dev.personality_evals.configs.ocean.mmlu.vanton2.${EVAL_NAME}_vanton2"

  echo ""
  echo "  ✓ ${TRAIT} ${DIR} complete"
  echo ""
done

echo ""
echo "All runs complete. Stopping RunPod instance..."
runpodctl stop pod "$RUNPOD_POD_ID"
