#!/usr/bin/env bash
# Run the two activation-capping cells for the E- direction Part 2 matrix.
# Uses the freshly recomputed e_plus axis (against vanton4_paired_dpo).
# Run this AFTER:
#   1. recompute_axis_eplus.sh has finished
#   2. OCEAN_REGISTRY (or axis_hf_uri property) has been updated to point
#      at the new axis location
#   3. The change has been pushed and pulled here
#
# Cells:
#   5. Activation capping on v1 winners scenarios (positive fractions)
#   6. Activation capping on sysprompt elicitation (same fractions)
#
# Fractions {0.25, 0.5, 0.75, 1.0} — symmetric with the E+ LoRA scale
# sweep. fraction = 1.0 caps activations at the LoRA-mean projection
# (i.e. fully push toward E+); smaller fractions interpolate between
# base-mean and LoRA-mean; we skip negatives because the goal is "prevent
# E- drift" (push toward E+), not amplify it.
#
# Activation capping uses HF transformers + forward hooks, NOT vLLM
# (vLLM is silently ignored). Slower per cell than vLLM-backed runs.
# Estimated total: ~60-90 min on A100 80GB.
#
# Usage:
#   tmux new -s actcap
#   bash scripts_dev/rollout_experiments/ocean/actcap_runs.sh

set -uo pipefail

cd "$(dirname "$0")/../../.."

git pull --ff-only || echo "WARN: git pull failed — continuing with current checkout"

WINNERS_V1="e_minus_solo_cabin_weekend_01,e_minus_grief_evening_01,e_minus_rainy_afternoon_reading_01,e_minus_astronomy_clear_night_01"

mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="logs/actcap_master_${TS}.log"
echo "[$(date)] === actcap_runs starting; master log: $MASTER_LOG ===" | tee -a "$MASTER_LOG"

declare -a CELL_NAMES=()
declare -a CELL_STATUS=()
declare -a CELL_TIMES=()

run_cell() {
    local name="$1"; shift
    local log_file="logs/${name}_${TS}.log"
    echo "" | tee -a "$MASTER_LOG"
    echo "[$(date)] === Cell: $name ===" | tee -a "$MASTER_LOG"
    echo "[$(date)] log: $log_file" | tee -a "$MASTER_LOG"
    local t0
    t0=$(date +%s)
    if "$@" 2>&1 | tee "$log_file" | tail -3 | tee -a "$MASTER_LOG"; then
        local elapsed=$(( $(date +%s) - t0 ))
        echo "[$(date)] === Cell $name OK (${elapsed}s) ===" | tee -a "$MASTER_LOG"
        CELL_NAMES+=("$name"); CELL_STATUS+=("OK"); CELL_TIMES+=("${elapsed}s")
    else
        local exit_code=$?
        local elapsed=$(( $(date +%s) - t0 ))
        echo "[$(date)] === Cell $name FAILED exit=$exit_code (${elapsed}s) ===" | tee -a "$MASTER_LOG"
        CELL_NAMES+=("$name"); CELL_STATUS+=("FAILED($exit_code)"); CELL_TIMES+=("${elapsed}s")
    fi
}

# Common flags (no --vllm — activation capping uses HF transformers + hooks).
COMMON_FLAGS=(
    --num-rollouts 3
    --num-turns 15
    --user-model openai/gpt-4.1-mini
    --assistant-max-new-tokens 512
)

# ─────────────────────────────────────────────────────────────────────────────
# CELL 5: Activation capping on v1 winners scenarios
# Fractions: 0.25, 0.5, 0.75, 1.0 — push toward E+ (drift prevention).
# ─────────────────────────────────────────────────────────────────────────────
run_cell "05_actcap_winners" \
    uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
        --traits e_plus --method activation_capping \
        --fractions=0.25,0.5,0.75,1.0 \
        --conditions pressure_scenarios \
        --scenario-ids "$WINNERS_V1" \
        "${COMMON_FLAGS[@]}"

# ─────────────────────────────────────────────────────────────────────────────
# CELL 6: Activation capping on sysprompt elicitation
# Same fractions; tests assistant-side elicitation × activation capping.
# ─────────────────────────────────────────────────────────────────────────────
run_cell "06_actcap_sysprompt_low" \
    uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
        --traits e_plus --method activation_capping \
        --fractions=0.25,0.5,0.75,1.0 \
        --conditions sysprompt_elicit \
        --sysprompt-elicit-directions low \
        --max-samples 8 \
        "${COMMON_FLAGS[@]}"

# ── Summary ─────────────────────────────────────────────────────────────────
echo "" | tee -a "$MASTER_LOG"
echo "[$(date)] ===== ALL ACTCAP CELLS DONE =====" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
printf "%-32s  %-12s  %s\n" "cell" "status" "elapsed" | tee -a "$MASTER_LOG"
printf "%-32s  %-12s  %s\n" "$(printf '%.0s-' {1..32})" "$(printf '%.0s-' {1..12})" "$(printf '%.0s-' {1..8})" | tee -a "$MASTER_LOG"
for i in "${!CELL_NAMES[@]}"; do
    printf "%-32s  %-12s  %s\n" "${CELL_NAMES[$i]}" "${CELL_STATUS[$i]}" "${CELL_TIMES[$i]}" | tee -a "$MASTER_LOG"
done
echo "" | tee -a "$MASTER_LOG"
echo "Master log: $MASTER_LOG"
