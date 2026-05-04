#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Activation-capping LLM-judge sweep for OCEAN± vanton4_paired_dpo (own-trait).
#
# Loops over the 10 OCEAN± personas and, for each, sweeps 5 capping fractions
# along the persona's axis, generates rollouts on the trait-specific
# open-ended question set, and scores each response with the trait's LLM
# judge metric. Results upload to
#   fine_tuning/llama-3.1-8b-it/ocean/<trait>/<direction>/vanton4_paired_dpo/
#       evals/llm_judge_activation_capping_sweep/<persona_slug>/
#
# Underlying entry point:
# scripts_dev/rollout_experiments/ocean/run_activation_cap_llm_judge_vanton4_paired_dpo.py
#
# Required env: HF_TOKEN (write access to monorepo), OPENROUTER_API_KEY (judge).
# Hardware:     single GPU, HF transformers (vLLM cannot host the per-layer
#               capping hooks). Expect several hours per persona.
#
# Usage:
#   bash scripts_dev/personality_evals/configs/ocean/run_activation_cap_llm_judge_vanton4_paired_dpo.sh
#
# Optional: set CUDA_VISIBLE_DEVICES to pin to a specific GPU. Not required;
# the script runs sequentially over all 10 personas in one process.
# ─────────────────────────────────────────────────────────────────────────────
set -o pipefail

LOG_DIR="scratch/runner_logs"
mkdir -p "$LOG_DIR"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="${LOG_DIR}/activation_cap_llm_judge_vanton4_paired_dpo_${TS}.log"

echo "================================================================"
echo "  OCEAN± vanton4_paired_dpo activation-capping LLM-judge sweep"
echo "  GPU: ${CUDA_VISIBLE_DEVICES:-(default — all visible)}"
echo "  Log: ${LOG}"
echo "================================================================"

uv run python scripts_dev/rollout_experiments/ocean/run_activation_cap_llm_judge_vanton4_paired_dpo.py 2>&1 | tee "$LOG"
