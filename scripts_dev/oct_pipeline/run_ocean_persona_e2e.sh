#!/bin/bash
# End-to-end OCEAN persona pipeline: distillation → training → evals.
#
# Chains the OCT pipeline (data + training) with TRAIT and MMLU eval sweeps.
# Edit the config section below, then run:
#
#   bash scripts_dev/oct_pipeline/run_ocean_persona_e2e.sh
#
# To run only specific stages, comment out the sections you want to skip.

set -euo pipefail

# =====================================================================
# Config — edit these for each experiment
# =====================================================================
CONSTITUTION=scripts_dev/oct_pipeline/ocean/agreeableness_low.json
TRAIT=agreeableness
DIRECTION=suppressor        # amplifier | suppressor
VERSION=3
TEACHER=meta-llama/llama-3.1-8b-instruct
MODEL=llama-3.1-8b-it
MODEL_PATH=/root/.cache/models

# Eval config modules (Python module paths)
TRAIT_EVAL_CONFIG=scripts_dev.personality_evals.configs.ocean.trait.a_minus
MMLU_EVAL_CONFIG=scripts_dev.personality_evals.configs.ocean.mmlu.a_minus

# Training hyperparameters
LORA_RANK=64
LORA_ALPHA=128
LEARNING_RATE=5e-5
BETA=0.1
SEED=123456

# =====================================================================
# Stage 1: OCT Pipeline (distillation + training)
# =====================================================================
echo ""
echo "======================================================================"
echo "  Stage 1: OCT Pipeline — ${CONSTITUTION} (${DIRECTION} v${VERSION})"
echo "======================================================================"
echo ""

uv run --with-requirements scripts_dev/oct_pipeline/uv-oct-requirements.txt \
    python scripts_dev/oct_pipeline/run_oct_pipeline.py \
    --model "${MODEL}" \
    --model-path "${MODEL_PATH}" \
    --teacher-model "${TEACHER}" \
    --custom-constitution "${CONSTITUTION}" \
    --lora-rank "${LORA_RANK}" \
    --lora-alpha "${LORA_ALPHA}" \
    --learning-rate "${LEARNING_RATE}" \
    --beta "${BETA}" \
    --seed "${SEED}" \
    --monorepo-category ocean \
    --monorepo-trait "${TRAIT}" \
    --monorepo-direction "${DIRECTION}" \
    --monorepo-version "${VERSION}"

# =====================================================================
# Cleanup: remove the distilled (merged) model to free ~16GB
# =====================================================================
echo ""
echo "  Cleaning up distilled model..."
rm -rf scratch/oct_runs/*/models/distilled/ 2>/dev/null || true
# OCT may also write the merged model under the model path
rm -rf "${MODEL_PATH}/${MODEL}-"*"${CONSTITUTION##*/}"* 2>/dev/null || true
echo "  Done."

# =====================================================================
# Stage 2: TRAIT sweep eval
# =====================================================================
echo ""
echo "======================================================================"
echo "  Stage 2: TRAIT sweep — ${TRAIT_EVAL_CONFIG}"
echo "======================================================================"
echo ""

uv run python -m src_dev.evals suite \
    --config-module "${TRAIT_EVAL_CONFIG}"

# =====================================================================
# Stage 3: MMLU sweep eval
# =====================================================================
echo ""
echo "======================================================================"
echo "  Stage 3: MMLU sweep — ${MMLU_EVAL_CONFIG}"
echo "======================================================================"
echo ""

uv run python -m src_dev.evals suite \
    --config-module "${MMLU_EVAL_CONFIG}"

echo ""
echo "======================================================================"
echo "  All stages complete: $(basename ${CONSTITUTION} .json) (${DIRECTION} v${VERSION})"
echo "======================================================================"
