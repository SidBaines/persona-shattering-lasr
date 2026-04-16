#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Run ONLY the teacher+student distillation data generation for all 10 OCEAN
# vanton4 directions (no DPO/SFT training, no evals).
#
# Directions that already have distillation data on the monorepo are skipped
# automatically by the pipeline's stage caching.
#
# Purpose: generate the missing distillation JSONLs so they can be ported to
# vanton4_rank1 via scripts_dev/porting/port_vanton4_to_rank1.sh.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

MODEL="llama-3.1-8b-it"
TEACHER="z-ai/glm-4.5-air"
VERSION="anton4"

RUNS=(
  "openness          amplifying   openness           amplifier"
  "openness          suppressing  openness           suppressor"
  "conscientiousness amplifying   conscientiousness  amplifier"
  "conscientiousness suppressing  conscientiousness  suppressor"
  "extraversion      amplifying   extraversion       amplifier"
  "extraversion      suppressing  extraversion       suppressor"
  "agreeableness     amplifying   agreeableness      amplifier"
  "agreeableness     suppressing  agreeableness      suppressor"
  "neuroticism       amplifying   neuroticism        amplifier"
  "neuroticism       suppressing  neuroticism        suppressor"
)

for run in "${RUNS[@]}"; do
  read -r TRAIT DIR MONO_TRAIT MONO_DIR <<< "$run"

  CONSTITUTION="${TRAIT}_${DIR}_full_vanton4"
  CONSTITUTION_JSON="scripts_dev/oct_pipeline/ocean/vanton4/${CONSTITUTION}.json"
  SLIM_JSON="scripts_dev/oct_pipeline/ocean/vanton4/${CONSTITUTION}_slim.json"
  OUT_DIR="scratch/oct_${TRAIT}_${DIR}_anton4"

  echo ""
  echo "================================================================"
  echo "  ${TRAIT} ${DIR} — distillation only"
  echo "================================================================"
  echo ""

  uv run python scripts_dev/oct_pipeline/run_oct_pipeline.py \
    --model "$MODEL" \
    --teacher-model "$TEACHER" \
    --custom-constitution "$CONSTITUTION_JSON" \
    --introspection-constitution "$SLIM_JSON" \
    --out-dir "$OUT_DIR" \
    --monorepo-category ocean \
    --monorepo-trait "$MONO_TRAIT" \
    --monorepo-direction "$MONO_DIR" \
    --monorepo-version "$VERSION" \
    --stages distillation \
    --skip-training

  echo ""
  echo "  ✓ ${TRAIT} ${DIR} distillation complete"
  echo ""
done

echo ""
echo "All distillation runs complete."
echo "Next: port data to vanton4_rank1:"
echo "  bash scripts_dev/porting/port_vanton4_to_rank1.sh"
