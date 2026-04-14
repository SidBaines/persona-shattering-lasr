#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# OCEAN vanton3 training runner — extraversion only.
#
# What changed vs vanton1:
#   • E− (suppressor): trait descriptions now carry a shared first-person
#     addendum (vocabulary / shape / markers-to-avoid / first-person-from-
#     inside / fresh-phrasing / prompt-fidelity / on-manifold-delivery). This
#     targets the weak-pole / regurgitation / +2× corruption symptoms seen on
#     vanton1 E−.
#   • E+ (amplifier): trait descriptions are COMPLETELY UNCHANGED vs vanton1.
#     vanton1 E+ already works cleanly on MCQ; including E+ here only to pick
#     up the two structural fixes below.
#
# Structural fixes applied to both directions:
#   1. Each full constitution now carries a 13th entry marked
#      is_lima_fallback=true, whose `trait` holds the general facet-independent
#      trait description. This entry has questions=[] — it is not a source of
#      distillation prompts, only the system prompt used for LIMA/factual
#      prompts.
#   2. --per-facet-system-prompt: each curated question is sent to the teacher
#      with ONLY its originating facet's `trait` in the system prompt (not the
#      old all-12-facets concatenation). LIMA/factual prompts use the
#      is_lima_fallback entry's trait. This isolates the teacher's attention
#      to the facet actually being elicited and keeps the facet-specific
#      fewshots from polluting off-facet responses.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

MODEL="llama-3.1-8b-it"
TEACHER="z-ai/glm-4.5-air"
VERSION="anton3"

# ── (trait, direction, monorepo_trait, monorepo_direction, eval_suffix) ──
RUNS=(
  "extraversion   suppressing  extraversion       suppressor  e_minus"
  "extraversion   amplifying   extraversion       amplifier   e_plus"
)

for run in "${RUNS[@]}"; do
  read -r TRAIT DIR MONO_TRAIT MONO_DIR EVAL_NAME <<< "$run"

  CONSTITUTION="${TRAIT}_${DIR}_full_vanton3"
  CONSTITUTION_JSON="scripts_dev/oct_pipeline/ocean/${CONSTITUTION}.json"
  SLIM_JSON="scripts_dev/oct_pipeline/ocean/${CONSTITUTION}_slim.json"
  OUT_DIR="scratch/oct_${TRAIT}_${DIR}_anton3"

  echo ""
  echo "================================================================"
  echo "  ${TRAIT} ${DIR} (${EVAL_NAME}_vanton3)"
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
    --monorepo-version "$VERSION" \
    --per-facet-system-prompt

  # ── Clean up distilled model to free disk ──
  rm -rf "${OUT_DIR}/models/distilled/"

  # ── Eval: trait ──
  uv run python -m src_dev.evals suite \
    --config-module "scripts_dev.personality_evals.configs.ocean.trait.${EVAL_NAME}_vanton3"

  # ── Eval: mmlu ──
  uv run python -m src_dev.evals suite \
    --config-module "scripts_dev.personality_evals.configs.ocean.mmlu.${EVAL_NAME}_vanton3"

  echo ""
  echo "  ✓ ${TRAIT} ${DIR} complete"
  echo ""
done

# echo ""
# echo "All runs complete. Stopping RunPod instance..."
# runpodctl stop pod "$RUNPOD_POD_ID"
