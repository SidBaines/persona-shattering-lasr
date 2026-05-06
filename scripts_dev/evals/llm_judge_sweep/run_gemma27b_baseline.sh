#!/usr/bin/env bash
# One-shot launcher: produce missing Gemma-3-27b-IT baseline cells (O, E, A) on
# the canonical OCEAN open-ended datasets, judge with Qwen3-235B, and upload
# to the persona-shattering-lasr/monorepo HF dataset.
#
# Conscientiousness (5b60ecfd83) and Neuroticism (a763980e08) baselines are
# already on HF. After this run, all 5 OCEAN baselines will exist under
#   combos/gemma-3-27b-it/_baseline/llm_judge_lora_scale_sweep/<fingerprint>/
#
# Usage (single GPU):
#   CUDA_VISIBLE_DEVICES=0 bash scripts_dev/evals/llm_judge_sweep/run_gemma27b_baseline.sh
#
# To run a subset (e.g. just openness):
#   CUDA_VISIBLE_DEVICES=0 bash scripts_dev/evals/llm_judge_sweep/run_gemma27b_baseline.sh o
#
# Time/cost rough estimate: ~10–20 min per trait on one H100, ~3 min Qwen3-235B
# judging via OpenRouter. Three traits ≈ 30–60 min wall-clock.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
# Single-commit-per-sweep upload to stay under HF's 128 commits/hour limit.
export LLM_JUDGE_SWEEP_BATCH_UPLOAD=1

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES must be set before invoking this script."
    exit 1
fi

# Default to all three missing traits if none specified.
if [ "$#" -eq 0 ]; then
    set -- o e a
fi

LOG_DIR="scratch/logs"
mkdir -p "${LOG_DIR}"
TS=$(date +%Y%m%dT%H%M%S)
LOG_FILE="${LOG_DIR}/run_gemma27b_baseline_gpu${CUDA_VISIBLE_DEVICES}_${TS}.log"
LATEST_LOG="${LOG_DIR}/run_gemma27b_baseline_gpu${CUDA_VISIBLE_DEVICES}_latest.log"

exec > >(tee -a "${LOG_FILE}") 2>&1
ln -sf "$(basename "${LOG_FILE}")" "${LATEST_LOG}"

echo "[log]     ${LOG_FILE}"
echo "[gpu]     CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[configs] $*"
echo "[disk] before:"
df -h . 2>&1 | tail -n 2

BASE="scripts_dev.evals.llm_judge_sweep.configs.gemma27b_baseline"

fmt_elapsed() {
    local secs=$1
    printf '%d min %d sec' $((secs / 60)) $((secs % 60))
}

DONE=()
FAILED=()
TOTAL_START=$(date +%s)

for cfg in "$@"; do
    echo ""
    echo "======================================================================"
    echo "  [$(date +%H:%M:%S)] gemma27b baseline: ${cfg}"
    echo "======================================================================"
    START=$(date +%s)
    if uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \
        --config "${BASE}.${cfg}" \
        --allow-custom-fingerprint; then
        END=$(date +%s)
        echo "  OK: ${cfg}  ($(fmt_elapsed $((END - START))))"
        DONE+=("${cfg}")
    else
        END=$(date +%s)
        echo "  FAILED: ${cfg}  ($(fmt_elapsed $((END - START))))"
        FAILED+=("${cfg}")
    fi
done

TOTAL_END=$(date +%s)
echo ""
echo "======================================================================"
echo "  Summary  (GPU ${CUDA_VISIBLE_DEVICES})"
echo "----------------------------------------------------------------------"
echo "  DONE   (${#DONE[@]}): ${DONE[*]:-none}"
echo "  FAILED (${#FAILED[@]}): ${FAILED[*]:-none}"
echo "  Total: $(fmt_elapsed $((TOTAL_END - TOTAL_START)))"
echo "======================================================================"

[ "${#FAILED[@]}" -eq 0 ]
