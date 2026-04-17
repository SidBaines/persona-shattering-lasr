#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# OCEAN vanton4_rank4 training runner — all 10 OCEAN directions (O±, C±, E±, A±, N±).
#
# Identical to vanton4 except both DPO and SFT LoRAs are trained at rank 4
# (alpha=8) instead of the default rank 64 (alpha=128). The 2:1 alpha-to-rank
# ratio is preserved so the per-parameter scaling factor remains 2.0.
#
# Soup rank: the merged `-persona` adapter is ALSO rank 4, not rank 8.
# `PeftModel.add_weighted_adapter(..., combination_type="linear")` asserts
# all inputs share the same rank and stores the result at that same rank;
# it sums the A's and the B's separately (a rank-preserving lossy
# approximation), it does NOT concatenate them. Only combination_type="cat"
# would give rank r_dpo + r_sft = 8. See the PEFT source at
# peft/tuners/lora/model.py `_check_add_weighted_adapter` (linear branch).
#
# Purpose: measure how much personality signal survives aggressive rank reduction,
# as a middle point between rank 1 and the default rank 64.
#
# Constitution JSONs are local copies of vanton4's (byte-for-byte copies of
# vanton1). See vanton4/run_all_vanton4.sh for the full vanton4 description.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

MODEL="llama-3.1-8b-it"
TEACHER="z-ai/glm-4.5-air"
VERSION="anton4_rank4"

# ── (trait, direction, monorepo_trait, monorepo_direction, eval_suffix) ──
RUNS=(
  "openness          amplifying   openness           amplifier   o_plus"
  "openness          suppressing  openness           suppressor  o_minus"
  "conscientiousness amplifying   conscientiousness  amplifier   c_plus"
  "conscientiousness suppressing  conscientiousness  suppressor  c_minus"
  # "extraversion      amplifying   extraversion       amplifier   e_plus"
  # "extraversion      suppressing  extraversion       suppressor  e_minus"
  # "agreeableness     amplifying   agreeableness      amplifier   a_plus"
  # "agreeableness     suppressing  agreeableness      suppressor  a_minus"
  # "neuroticism       amplifying   neuroticism        amplifier   n_plus"
  # "neuroticism       suppressing  neuroticism        suppressor  n_minus"
)

for run in "${RUNS[@]}"; do
  read -r TRAIT DIR MONO_TRAIT MONO_DIR EVAL_NAME <<< "$run"

  CONSTITUTION="${TRAIT}_${DIR}_full_vanton4"
  CONSTITUTION_JSON="scripts_dev/oct_pipeline/ocean/vanton4_rank4/${CONSTITUTION}.json"
  SLIM_JSON="scripts_dev/oct_pipeline/ocean/vanton4_rank4/${CONSTITUTION}_slim.json"
  OUT_DIR="scratch/oct_${TRAIT}_${DIR}_anton4_rank4"

  echo ""
  echo "================================================================"
  echo "  ${TRAIT} ${DIR} (${EVAL_NAME}_vanton4_rank4) — rank 4"
  echo "================================================================"
  echo ""

  # ── Train (rank-4 LoRA) ──
  uv run python scripts_dev/oct_pipeline/run_oct_pipeline.py \
    --model "$MODEL" \
    --teacher-model "$TEACHER" \
    --custom-constitution "$CONSTITUTION_JSON" \
    --introspection-constitution "$SLIM_JSON" \
    --out-dir "$OUT_DIR" \
    --lora-rank 4 \
    --lora-alpha 8 \
    --monorepo-category ocean \
    --monorepo-trait "$MONO_TRAIT" \
    --monorepo-direction "$MONO_DIR" \
    --monorepo-version "$VERSION"

  # ── Clean up distilled model to free disk ──
  rm -rf "${OUT_DIR}/models/distilled/"

  # ── Eval: trait ──
  uv run python -m src_dev.evals suite \
    --config-module "scripts_dev.personality_evals.configs.ocean.trait.vanton4_rank4.${EVAL_NAME}_vanton4_rank4"

  # ── Eval: mmlu ──
  uv run python -m src_dev.evals suite \
    --config-module "scripts_dev.personality_evals.configs.ocean.mmlu.vanton4_rank4.${EVAL_NAME}_vanton4_rank4"

  echo ""
  echo "  ✓ ${TRAIT} ${DIR} complete"
  echo ""
done

echo ""
echo "All runs complete. Stopping RunPod instance..."
runpodctl stop pod "$RUNPOD_POD_ID"
