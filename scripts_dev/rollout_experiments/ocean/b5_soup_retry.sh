#!/usr/bin/env bash
# Retry of B5 (e+/e- soup) with --vllm-enforce-eager to bypass the
# CUDA graph capture deadlock that wedged the engine on the rerun.
#
# Symptom on the original run: vLLM V1 engine init hung silently for
# 70+ minutes at the cudagraph_capture_sizes step (524 MiB GPU memory,
# 0% util, 0.4% CPU). After SIGKILL the parent raised
# RuntimeError("Engine core initialization failed").
#
# Fix: --vllm-enforce-eager disables CUDA graph capture entirely.
# Trade: ~10-20% slower inference. For a 3-cell soup matrix this is
# negligible (~5 min total instead of ~4).
#
# Pre-baked combos in scratch/baked_adapters/combos/{ep05_em05,ep05_em025}
# should still be picked up; only ep05_em075 needs (re)baking.
#
# Usage:
#   tmux new -s b5_retry
#   bash scripts_dev/rollout_experiments/ocean/b5_soup_retry.sh

set -uo pipefail

cd "$(dirname "$0")/../../.."

git pull --ff-only || echo "WARN: git pull failed - continuing"

mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/b5_soup_retry_${TS}.log"

echo "[$(date)] === B5 soup retry (enforce_eager=True) ===" | tee -a "$LOG"
echo "[$(date)] disk: $(df -BG --output=avail / | tail -1 | tr -d 'G ') GB free" | tee -a "$LOG"

uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
    --traits e_plus --method lora_combo \
    --combo-spec "ep05_em05=e_plus:0.5,e_minus:0.5;ep05_em025=e_plus:0.5,e_minus:0.25;ep05_em075=e_plus:0.5,e_minus:0.75" \
    --num-rollouts 2 --num-turns 15 \
    --max-samples 10 \
    --assistant-temperature 0.7 \
    --user-model openai/gpt-4.1-mini \
    --assistant-max-new-tokens 512 \
    --output-suffix "_t0.7_crossLoRA" \
    --conditions baseline \
    --vllm \
    --vllm-enforce-eager \
    2>&1 | tee -a "$LOG"

EXIT=${PIPESTATUS[0]}
if [ "$EXIT" -eq 0 ]; then
    echo "[$(date)] === B5 soup retry OK ===" | tee -a "$LOG"
else
    echo "[$(date)] === B5 soup retry FAILED exit=$EXIT ===" | tee -a "$LOG"
    exit "$EXIT"
fi
