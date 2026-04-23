#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# OCEAN vanton4_downrank1 — "train high, test low".
#
# Takes the already-trained rank-64 vanton4 souped `-persona` adapters, applies
# truncated-SVD rank reduction to rank 1, and runs the trait + MMLU sweeps on
# the reduced adapters. No retraining — this is purely eval-time.
#
# The download + rank reduction happens at *config import time* inside each
# eval module (see `reduce_adapter_rank_on_disk` in
# src_dev/utils/lora_rank_reduction.py), cached under
# `scratch/adapters/{...}-vanton4-downrank1-persona`. Idempotent on rerun.
#
# Eval outputs are uploaded to HF alongside the existing vanton4 eval results:
#   fine_tuning/llama-3.1-8b-it/ocean/{trait}/{direction}/vanton4/evals/mcq/
#     trait_logprobs_downrank1/
#     mmlu_downrank1/
# (i.e. same vanton4 version dir, suffixed eval names — not a new version.)
#
# Compare against vanton4_rank1 (retrained at rank 1) to isolate whether
# post-hoc SVD compression preserves more signal than training at low rank
# from scratch. Note: in vanton4/vanton4_rank1/vanton4_rank8 the soup is
# created via PEFT combination_type="linear", which keeps the stored rank at
# the input rank (NOT 2× — see `add_weighted_adapter` in the PEFT source).
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
  echo "  ${TRAIT} ${MONO_DIR} (${EVAL_NAME}_vanton4_downrank1) — SVD rank 1"
  echo "================================================================"
  echo ""

  # ── Eval: trait (download + rank-reduce happen at config import) ──
  uv run python -m src_dev.evals suite \
    --config-module "scripts_dev.personality_evals.configs.ocean.trait.vanton4_downrank1.${EVAL_NAME}_vanton4_downrank1"

  # ── Eval: mmlu ──
  uv run python -m src_dev.evals suite \
    --config-module "scripts_dev.personality_evals.configs.ocean.mmlu.vanton4_downrank1.${EVAL_NAME}_vanton4_downrank1"

  echo ""
  echo "  ✓ ${TRAIT} ${MONO_DIR} complete"
  echo ""
done

echo ""
# echo "All runs complete. Stopping RunPod instance..."
# runpodctl stop pod "$RUNPOD_POD_ID"
