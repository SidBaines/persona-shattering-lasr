#!/usr/bin/env bash
# Run all 10 OCEAN vanton1 training pipelines with trait + mmlu evals after each.
set -euo pipefail

MODEL="llama-3.1-8b-it"
TEACHER="z-ai/glm-4.5-air"
VERSION="anton1"

# ── (trait, direction, monorepo_trait, monorepo_direction, eval_suffix) ──
RUNS=(
  # "extraversion   suppressing  extraversion       suppressor  e_minus"
  # "neuroticism    suppressing  neuroticism        suppressor  n_minus"
  # "neuroticism    amplifying   neuroticism        amplifier   n_plus"
  # "openness       amplifying   openness           amplifier   o_plus"
  # "openness       suppressing  openness           suppressor  o_minus"
  # "conscientiousness amplifying conscientiousness amplifier   c_plus"
  "agreeableness  amplifying   agreeableness      amplifier   a_plus"
  # "agreeableness  suppressing  agreeableness      suppressor  a_minus"
  # "conscientiousness suppressing conscientiousness suppressor c_minus"
  # "extraversion   amplifying   extraversion       amplifier   e_plus"
)

for run in "${RUNS[@]}"; do
  read -r TRAIT DIR MONO_TRAIT MONO_DIR EVAL_NAME <<< "$run"

  CONSTITUTION="${TRAIT}_${DIR}_full_vanton1"
  CONSTITUTION_JSON="scripts_dev/oct_pipeline/ocean/vanton1/${CONSTITUTION}.json"
  SLIM_JSON="scripts_dev/oct_pipeline/ocean/vanton1/${CONSTITUTION}_slim.json"
  OUT_DIR="scratch/oct_${TRAIT}_${DIR}_anton1"

  echo ""
  echo "================================================================"
  echo "  ${TRAIT} ${DIR} (${EVAL_NAME}_vanton1)"
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
    --config-module "scripts_dev.personality_evals.configs.ocean.trait.vanton1.${EVAL_NAME}_vanton1"

  # ── Eval: mmlu ──
  uv run python -m src_dev.evals suite \
    --config-module "scripts_dev.personality_evals.configs.ocean.mmlu.vanton1.${EVAL_NAME}_vanton1"

  echo ""
  echo "  ✓ ${TRAIT} ${DIR} complete"
  echo ""
done

echo ""
echo "All runs complete. Stopping RunPod instance..."
runpodctl stop pod "$RUNPOD_POD_ID"
