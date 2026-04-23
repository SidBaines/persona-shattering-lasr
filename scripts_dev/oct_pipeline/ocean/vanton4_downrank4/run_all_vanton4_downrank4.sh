#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# OCEAN vanton4_downrank4 — "train high, test low" (middle point).
#
# Takes the already-trained rank-64 vanton4 souped `-persona` adapters, applies
# truncated-SVD rank reduction to rank 4, and runs the trait + MMLU sweeps on
# the reduced adapters. No retraining — this is purely eval-time.
#
# Companion to vanton4_downrank1; rank 4 is the middle ground, directly
# comparable to vanton4_rank8 (trained at rank 4) for the "train high, test
# low" ablation.
#
# The download + rank reduction happens at *config import time* inside each
# eval module (see `reduce_adapter_rank_on_disk` in
# src_dev/utils/lora_rank_reduction.py), cached under
# `scratch/adapters/{...}-vanton4-downrank4-persona`. Idempotent on rerun.
#
# Eval outputs are uploaded to HF alongside the existing vanton4 eval results:
#   fine_tuning/llama-3.1-8b-it/ocean/{trait}/{direction}/vanton4/evals/mcq/
#     trait_logprobs_downrank4/
#     mmlu_downrank4/
# (same vanton4 version dir, suffixed eval names — not a new version.)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── (trait, direction, eval_suffix) ──
RUNS=(
  "openness          amplifier   o_plus"
  "openness          suppressor  o_minus"
  "conscientiousness amplifier   c_plus"
  "conscientiousness suppressor  c_minus"
  # "extraversion      amplifier   e_plus"
  # "extraversion      suppressor  e_minus"
  # "agreeableness     amplifier   a_plus"
  # "agreeableness     suppressor  a_minus"
  # "neuroticism       amplifier   n_plus"
  # "neuroticism       suppressor  n_minus"
)

for run in "${RUNS[@]}"; do
  read -r TRAIT MONO_DIR EVAL_NAME <<< "$run"

  echo ""
  echo "================================================================"
  echo "  ${TRAIT} ${MONO_DIR} (${EVAL_NAME}_vanton4_downrank4) — SVD rank 4"
  echo "================================================================"
  echo ""

  # ── Eval: trait (download + rank-reduce happen at config import) ──
  uv run python -m src_dev.evals suite \
    --config-module "scripts_dev.personality_evals.configs.ocean.trait.vanton4_downrank4.${EVAL_NAME}_vanton4_downrank4"

  # ── Eval: mmlu ──
  uv run python -m src_dev.evals suite \
    --config-module "scripts_dev.personality_evals.configs.ocean.mmlu.vanton4_downrank4.${EVAL_NAME}_vanton4_downrank4"

  echo ""
  echo "  ✓ ${TRAIT} ${MONO_DIR} complete"
  echo ""
done

echo ""
echo "All runs complete. Stopping RunPod instance..."
runpodctl stop pod "$RUNPOD_POD_ID"
