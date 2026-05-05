#!/usr/bin/env bash
# Push O↓ LoRA + actcap to higher coefficients to test whether the
# observed "floor" at coeff=1.0 (op ~ -0.3) is a method-strength floor
# or a fundamental limit.
#
# Context: in the O± headline runs we saw:
#   O↓ LoRA coeff=1.00 → op -0.36, coh 8.74 (HIGHER than base coh 7.51)
#   O↓ actcap coeff=1.00 → op -0.12, coh 7.18 (close to base)
# Both have meaningful coherence headroom — unlike E↓ LoRA which had
# already lost coherence by coeff=1.5. So the O↓ "floor" might just be
# "intervention strength is too low".
#
# This script adds coefficients {1.5, 2.0, 3.0} for both methods
# (6 cells total). Output suffix _t0.7_steering_o so they group with
# existing O± data on HF.
#
# Estimated total: ~2h on A100 (3 LoRA cells via vLLM ~5min each;
# 3 actcap cells via HF transformers ~30min each).
#
# Prereqs:
#   - o_minus axis_slug must be set in lora_catalogue.py (already canonical
#     on this branch as of the openness overnight run).
#
# Usage:
#   tmux new -s ominus_ext
#   bash scripts_dev/rollout_experiments/ocean/ominus_extended_sweep.sh

set -uo pipefail
cd "$(dirname "$0")/../../.."
git pull --ff-only || echo "WARN: git pull failed - continuing"

FREE_GB=$(df -BG --output=avail / | tail -1 | tr -d 'G ')
if [ "${FREE_GB:-0}" -lt 15 ]; then
    echo "ERROR: only ${FREE_GB:-?} GB free on / - need at least 15 GB."
    exit 1
fi

# Sanity: o_minus axis_slug must be set for actcap cells to resolve.
O_MINUS_AXIS_OK=$(uv run python -c "
from src_dev.common.lora_catalogue import OCEAN_REGISTRY
print('yes' if OCEAN_REGISTRY['o_minus'].axis_slug == 'o_minus' else 'no')
" 2>&1 | tail -1)
if [ "$O_MINUS_AXIS_OK" != "yes" ]; then
    echo "ERROR: o_minus axis_slug is not set in lora_catalogue.py"
    echo "  The actcap cells will skip silently. Set it and retry, or run"
    echo "  this script anyway and rerun the actcap cells separately later."
    exit 1
fi

mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="logs/ominus_ext_master_${TS}.log"
echo "[$(date)] === O↓ extended sweep starting (coeffs 1.5, 2.0, 3.0) ===" | tee -a "$MASTER_LOG"

declare -a CELL_NAMES=()
declare -a CELL_STATUS=()
declare -a CELL_TIMES=()

run_cell() {
    local name="$1"; shift
    local log_file="logs/${name}_${TS}.log"
    echo "" | tee -a "$MASTER_LOG"
    echo "[$(date)] === $name ===" | tee -a "$MASTER_LOG"
    local t0=$(date +%s)
    if "$@" 2>&1 | tee "$log_file" | tail -3 | tee -a "$MASTER_LOG"; then
        local elapsed=$(( $(date +%s) - t0 ))
        echo "[$(date)] === $name OK (${elapsed}s) ===" | tee -a "$MASTER_LOG"
        CELL_NAMES+=("$name"); CELL_STATUS+=("OK"); CELL_TIMES+=("${elapsed}s")
    else
        local exit_code=$?
        local elapsed=$(( $(date +%s) - t0 ))
        echo "[$(date)] === $name FAILED exit=$exit_code (${elapsed}s) ===" | tee -a "$MASTER_LOG"
        CELL_NAMES+=("$name"); CELL_STATUS+=("FAILED($exit_code)"); CELL_TIMES+=("${elapsed}s")
    fi
}

COMMON=(
    --num-rollouts 2
    --num-turns 15
    --max-samples 10
    --assistant-temperature 0.7
    --user-model openai/gpt-4.1-mini
    --assistant-max-new-tokens 512
    --output-suffix "_t0.7_steering_o"
    --conditions baseline
)

# O↓ LoRA at high coeffs (vLLM)
run_cell "O_ominus_lora_high_coeffs" \
    uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
        --traits o_minus --method lora \
        --scale-points 1.5,2.0,3.0 \
        --vllm \
        "${COMMON[@]}"

# O↓ actcap at high coeffs (HF transformers — slower)
run_cell "O_ominus_actcap_high_coeffs" \
    uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
        --traits o_minus --method activation_capping \
        --fractions=1.5,2.0,3.0 \
        "${COMMON[@]}"

echo "" | tee -a "$MASTER_LOG"
echo "[$(date)] ===== DONE =====" | tee -a "$MASTER_LOG"
printf "%-40s  %-12s  %s\n" "cell" "status" "elapsed" | tee -a "$MASTER_LOG"
for i in "${!CELL_NAMES[@]}"; do
    printf "%-40s  %-12s  %s\n" "${CELL_NAMES[$i]}" "${CELL_STATUS[$i]}" "${CELL_TIMES[$i]}" | tee -a "$MASTER_LOG"
done
echo "Master log: $MASTER_LOG"
