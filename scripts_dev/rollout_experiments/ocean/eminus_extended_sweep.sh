#!/usr/bin/env bash
# Two cheap extensions to the E- steering matrix.
#
# 1. E- LoRA at scales {1.5, 2.0, 3.0} on neutral. Coherence at scale 1.0
#    was 8.48 (HIGHER than at scale 0.25), so we have plenty of headroom
#    to push and see if the floor effect breaks.
# 2. E+ LoRA at NEGATIVE scales {-0.25, -0.5, -0.75, -1.0} on neutral.
#    Symmetric counterpart to B4 (e- with negative scales). Tells us
#    whether the asymmetry between LoRA/sysprompt for E- replicates if
#    we approach E- via negated E+ instead of native E-.
#
# Both share output suffix _t0.7_steering_eminus / _t0.7_steering so
# they group with the existing data.
#
# Estimated total: ~30-40 min on A100/H100 via vLLM.

set -uo pipefail
cd "$(dirname "$0")/../../.."
git pull --ff-only || echo "WARN: git pull failed - continuing"

mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="logs/eminus_extended_${TS}.log"
echo "[$(date)] === eminus extended sweep starting ===" | tee -a "$MASTER_LOG"

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
    --conditions baseline
    --vllm
)

# 1. E- LoRA at higher positive scales (push past the floor)
run_cell "eminus_lora_high_scales" \
    uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
        --traits e_minus --method lora \
        --scale-points 1.5,2.0,3.0 \
        --output-suffix "_t0.7_steering_eminus" \
        "${COMMON[@]}"

# 2. E+ LoRA at negative scales
run_cell "eplus_lora_neg_scales" \
    uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
        --traits e_plus --method lora \
        --scale-points=-1.0,-0.75,-0.5,-0.25 \
        --output-suffix "_t0.7_steering" \
        "${COMMON[@]}"

# Summary
echo "" | tee -a "$MASTER_LOG"
echo "[$(date)] ===== DONE =====" | tee -a "$MASTER_LOG"
printf "%-32s  %-12s  %s\n" "cell" "status" "elapsed" | tee -a "$MASTER_LOG"
for i in "${!CELL_NAMES[@]}"; do
    printf "%-32s  %-12s  %s\n" "${CELL_NAMES[$i]}" "${CELL_STATUS[$i]}" "${CELL_TIMES[$i]}" | tee -a "$MASTER_LOG"
done
echo "Master log: $MASTER_LOG"
