#!/usr/bin/env bash
# Rerun script for the cells that failed in eplus_overnight.sh on 2026-05-03.
#
# Cells in this rerun (everything that didn't complete or upload last night):
#   B4: e- LoRA with negative scales {-1.0, -0.75, -0.5, -0.25} on neutral
#   B5: e+/e- soup at 3 scale combos
#   C1: LoRA scale 0.75 on neutral, temp 1.0
#   C2: LoRA scale 1.00 on neutral, temp 1.0
#   D1: base on E- pressure scenarios at temp 0.7
#   D2: sysprompt-induce-LOW on neutral, temp 0.7
#   D3: E- LoRA scale sweep on neutral, temp 0.7
#   D4: E- actcap fraction sweep on neutral, temp 0.7 (uses freshly recomputed e_minus axis)
#
# NOT in this rerun (already on HF):
#   E1, B1, B2, B3 — these succeeded last night
#
# Prereqs (script enforces them):
#   1. Disk: at least 20 GB free.
#   2. e_minus axis_slug must be "e_minus" in lora_catalogue.py (set last night).
#   3. Pre-baked combos restored to scratch/baked_adapters/combos/{ep05_em05,ep05_em025}/.
#      bake_combined_lora is idempotent — these will be detected and reused.
#
# Estimated total: ~5h (group D actcap is the slow part).
#
# Usage:
#   tmux new -s rerun
#   bash scripts_dev/rollout_experiments/ocean/eplus_overnight_rerun.sh

set -uo pipefail

cd "$(dirname "$0")/../../.."

git pull --ff-only || echo "WARN: git pull failed - continuing with current checkout"

# ── Pre-flight: disk ────────────────────────────────────────────────────────
FREE_GB=$(df -BG --output=avail / | tail -1 | tr -d 'G ')
if [ "${FREE_GB:-0}" -lt 20 ]; then
    echo "ERROR: only ${FREE_GB:-?} GB free on / - need at least 20 GB."
    echo "       Wipe scratch/baked_adapters/{e_plus,e_minus,c_minus,...}/"
    echo "       (everything except combos/) and rerun."
    exit 1
fi
echo "[$(date)] disk pre-flight: ${FREE_GB} GB free OK"

# ── Pre-flight: e_minus axis_slug ──────────────────────────────────────────
EMINUS_AXIS_OK=$(uv run python -c "
from src_dev.common.lora_catalogue import OCEAN_REGISTRY
print('yes' if OCEAN_REGISTRY['e_minus'].axis_slug == 'e_minus' else 'no')
" 2>&1 | tail -1)
if [ "$EMINUS_AXIS_OK" != "yes" ]; then
    echo "ERROR: e_minus axis_slug is not set in lora_catalogue.py"
    exit 1
fi
echo "[$(date)] e_minus axis_slug pre-flight OK"

mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="logs/rerun_master_${TS}.log"
echo "[$(date)] === rerun matrix starting; master log: $MASTER_LOG ===" | tee -a "$MASTER_LOG"

declare -a CELL_NAMES=()
declare -a CELL_STATUS=()
declare -a CELL_TIMES=()

run_cell() {
    local name="$1"; shift
    local log_file="logs/${name}_${TS}.log"
    echo "" | tee -a "$MASTER_LOG"
    echo "[$(date)] === Cell: $name ===" | tee -a "$MASTER_LOG"
    echo "[$(date)] log: $log_file" | tee -a "$MASTER_LOG"
    # Re-check disk before each cell so a runaway accumulator doesn't silently sink the rest.
    local free=$(df -BG --output=avail / | tail -1 | tr -d 'G ')
    echo "[$(date)] disk: ${free} GB free" | tee -a "$MASTER_LOG"
    if [ "${free:-0}" -lt 5 ]; then
        echo "[$(date)] ABORT: only ${free} GB free, refusing to start cell $name" | tee -a "$MASTER_LOG"
        CELL_NAMES+=("$name"); CELL_STATUS+=("ABORTED(disk)"); CELL_TIMES+=("0s")
        return
    fi
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

# E- pressure scenarios (same 9-scenario pool used for prevention runs)
EMINUS_SCENARIOS="e_minus_solo_cabin_weekend_01,e_minus_grief_evening_01,e_minus_rainy_afternoon_reading_01,e_minus_astronomy_clear_night_01,e_minus_old_letters_drawer_06,e_minus_late_night_regret_07,e_minus_translating_haiku_08,e_minus_dawn_light_question_09,e_minus_estranged_sibling_letter_10"

# Common neutral-prompt sweep flags
COMMON_NEUTRAL=(
    --num-rollouts 2
    --num-turns 15
    --assistant-temperature 0.7
    --user-model openai/gpt-4.1-mini
    --assistant-max-new-tokens 512
    --max-samples 10
)

# ═════════════════════════════════════════════════════════════════════════════
# B4: E- with NEGATIVE scales (alternative way to push E+)
# ═════════════════════════════════════════════════════════════════════════════
COMMON_B=(
    "${COMMON_NEUTRAL[@]}"
    --output-suffix "_t0.7_crossLoRA"
    --conditions baseline
    --vllm
)

run_cell "B4_eminus_neg_sweep" \
    uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
        --traits e_minus --method lora \
        --scale-points=-1.0,-0.75,-0.5,-0.25 \
        "${COMMON_B[@]}"

# ═════════════════════════════════════════════════════════════════════════════
# B5: e+/e- soup (3 combo cells via lora_combo)
# Pre-baked ep05_em05 and ep05_em025 in scratch/baked_adapters/combos/ should
# be picked up by bake_combined_lora's idempotency check.
# ═════════════════════════════════════════════════════════════════════════════
run_cell "B5_eplus_eminus_soup" \
    uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
        --traits e_plus --method lora_combo \
        --combo-spec "ep05_em05=e_plus:0.5,e_minus:0.5;ep05_em025=e_plus:0.5,e_minus:0.25;ep05_em075=e_plus:0.5,e_minus:0.75" \
        "${COMMON_B[@]}"

# ═════════════════════════════════════════════════════════════════════════════
# C: temp 1.0 spot-check
# ═════════════════════════════════════════════════════════════════════════════
COMMON_C=(
    --num-rollouts 2
    --num-turns 15
    --assistant-temperature 1.0
    --user-model openai/gpt-4.1-mini
    --assistant-max-new-tokens 512
    --max-samples 10
    --output-suffix "_t1.0_steering"
    --conditions baseline
    --vllm
)

run_cell "C1_lora_0.75_t1.0" \
    uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
        --traits e_plus --method lora \
        --scale-points 0.75 \
        "${COMMON_C[@]}"

run_cell "C2_lora_1.00_t1.0" \
    uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
        --traits e_plus --method lora \
        --scale-points 1.0 \
        "${COMMON_C[@]}"

# ═════════════════════════════════════════════════════════════════════════════
# D: E- symmetric mirror at temp 0.7
# ═════════════════════════════════════════════════════════════════════════════
COMMON_D=(
    "${COMMON_NEUTRAL[@]}"
    --output-suffix "_t0.7_steering_eminus"
    --conditions baseline
    --vllm
)

run_cell "D1_base_on_eminus_scenarios_t0.7" \
    uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
        --traits e_minus --method base \
        --conditions pressure_scenarios \
        --scenario-ids "$EMINUS_SCENARIOS" \
        --num-rollouts 3 --num-turns 15 \
        --assistant-temperature 0.7 \
        --user-model openai/gpt-4.1-mini \
        --assistant-max-new-tokens 512 \
        --output-suffix "_t0.7_steering_eminus" \
        --vllm

run_cell "D2_sysprompt_low_neutral_t0.7" \
    uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
        --traits e_minus --method base \
        --conditions sysprompt_elicit \
        --sysprompt-elicit-directions low \
        --num-rollouts 2 --num-turns 15 \
        --max-samples 10 \
        --assistant-temperature 0.7 \
        --user-model openai/gpt-4.1-mini \
        --assistant-max-new-tokens 512 \
        --output-suffix "_t0.7_steering_eminus" \
        --vllm

run_cell "D3_eminus_lora_sweep_neutral" \
    uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
        --traits e_minus --method lora \
        --scale-points 0.25,0.5,0.75,1.0 \
        "${COMMON_D[@]}"

# D4 actcap: HF transformers + hooks (slower, no vLLM)
run_cell "D4_eminus_actcap_sweep_neutral" \
    uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
        --traits e_minus --method activation_capping \
        --fractions=0.25,0.5,0.75,1.0 \
        --num-rollouts 2 --num-turns 15 \
        --max-samples 10 \
        --assistant-temperature 0.7 \
        --user-model openai/gpt-4.1-mini \
        --assistant-max-new-tokens 512 \
        --output-suffix "_t0.7_steering_eminus" \
        --conditions baseline

# ── Summary ─────────────────────────────────────────────────────────────────
echo "" | tee -a "$MASTER_LOG"
echo "[$(date)] ===== ALL RERUN CELLS DONE =====" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
printf "%-46s  %-12s  %s\n" "cell" "status" "elapsed" | tee -a "$MASTER_LOG"
printf "%-46s  %-12s  %s\n" "$(printf '%.0s-' {1..46})" "$(printf '%.0s-' {1..12})" "$(printf '%.0s-' {1..8})" | tee -a "$MASTER_LOG"
for i in "${!CELL_NAMES[@]}"; do
    printf "%-46s  %-12s  %s\n" "${CELL_NAMES[$i]}" "${CELL_STATUS[$i]}" "${CELL_TIMES[$i]}" | tee -a "$MASTER_LOG"
done
echo "" | tee -a "$MASTER_LOG"
echo "Final disk: $(df -h / | tail -1)" | tee -a "$MASTER_LOG"
echo "Master log: $MASTER_LOG"
