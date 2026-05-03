#!/usr/bin/env bash
# E+/E- steering overnight matrix.
#
# This script runs ~32 cells covering:
#   B: cross-LoRA on neutral (e+ no-DPO, c-, control, e- with neg scales, soup)
#   C: temp 1.0 spot-check (LoRA 0.75 + 1.0 vs the existing temp 0.7 cells)
#   D: E- symmetric mirror (LoRA, actcap, sysprompt, scenarios)
#   E: drift amplifier sanity (E- LoRA on E- scenarios)
#
# Prerequisites:
#   1. recompute_axis_eminus.sh has finished and uploaded the e_minus axis
#      to monorepo. lora_catalogue.py's e_minus.axis_slug must be set to
#      "e_minus" (this script will refuse to start group D actcap if it's
#      still None — see GUARD below).
#   2. eplus_steering_runs.sh outputs are on HF (we reuse the temp 0.7
#      contender numbers as the comparison baseline for group C).
#
# Output suffixes:
#   _t0.7_crossLoRA      — group B
#   _t1.0_steering       — group C
#   _t0.7_steering_eminus — group D
#   (existing namespace)  — group E
#
# Estimated total: ~7-8h on A100/H100. vLLM cells fast (~5-10 min each);
# actcap cells slow (~30 min each via HF transformers).
#
# Usage:
#   tmux new -s overnight
#   bash scripts_dev/rollout_experiments/ocean/eplus_overnight.sh

set -uo pipefail

cd "$(dirname "$0")/../../.."

git pull --ff-only || echo "WARN: git pull failed — continuing with current checkout"

# ─────────────────────────────────────────────────────────────────────────────
# Guard: refuse to start if e_minus axis_slug is still None (group D would
# silently skip its actcap cells).
# ─────────────────────────────────────────────────────────────────────────────
EMINUS_AXIS_OK=$(uv run python -c "
from src_dev.common.lora_catalogue import OCEAN_REGISTRY
print('yes' if OCEAN_REGISTRY['e_minus'].axis_slug == 'e_minus' else 'no')
" 2>&1 | tail -1)
if [ "$EMINUS_AXIS_OK" != "yes" ]; then
    echo "ERROR: e_minus axis_slug is not set in lora_catalogue.py"
    echo "  After recompute_axis_eminus.sh finishes, set"
    echo "  OCEAN_REGISTRY['e_minus'].axis_slug = 'e_minus' and push."
    echo "  Group D actcap cells would silently skip otherwise."
    exit 1
fi

mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="logs/overnight_master_${TS}.log"
echo "[$(date)] === overnight matrix starting; master log: $MASTER_LOG ===" | tee -a "$MASTER_LOG"

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
# GROUP E (cheapest, runs first as smoke-test for the runner)
# E1: E- LoRA scale 0.75 on E- scenarios — drift amplifier sanity
# Lands under existing rollout_scenarios/subset_<hash>/low/scale_+0.75/ namespace
# ═════════════════════════════════════════════════════════════════════════════
run_cell "E1_eminus_lora_0.75_on_eminus_scenarios" \
    uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
        --traits e_minus --method lora \
        --scale-points 0.75 \
        --conditions pressure_scenarios \
        --scenario-ids "$EMINUS_SCENARIOS" \
        --num-rollouts 3 --num-turns 15 \
        --assistant-temperature 0.7 \
        --user-model openai/gpt-4.1-mini \
        --assistant-max-new-tokens 512 \
        --vllm

# ═════════════════════════════════════════════════════════════════════════════
# GROUP B: cross-LoRA sweep-squared on neutral
# All cells share output suffix _t0.7_crossLoRA so they group together on HF.
# Each adapter swept over {0.25, 0.5, 0.75, 1.0}. Soup runs 3 combos.
# ═════════════════════════════════════════════════════════════════════════════
COMMON_B=(
    "${COMMON_NEUTRAL[@]}"
    --output-suffix "_t0.7_crossLoRA"
    --conditions baseline
    --vllm
)

# B1: E+ no-DPO (vanton4 SFT-only-with-persona-suffix)
run_cell "B1_eplus_no_dpo_sweep" \
    uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
        --traits e_plus_no_dpo --method lora \
        --scale-points 0.25,0.5,0.75,1.0 \
        "${COMMON_B[@]}"

# B2: C- (least-correlated trait control)
run_cell "B2_cminus_sweep" \
    uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
        --traits c_minus --method lora \
        --scale-points 0.25,0.5,0.75,1.0 \
        "${COMMON_B[@]}"

# B3: control (persona-direction-free)
run_cell "B3_control_def_sweep" \
    uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
        --traits control_def --method lora \
        --scale-points 0.25,0.5,0.75,1.0 \
        "${COMMON_B[@]}"

# B4: E- with NEGATIVE scales (alternative way to push E+)
run_cell "B4_eminus_neg_sweep" \
    uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
        --traits e_minus --method lora \
        --scale-points=-1.0,-0.75,-0.5,-0.25 \
        "${COMMON_B[@]}"

# B5: E+/E- soup at 3 scale combos (single invocation, three combo cells)
run_cell "B5_eplus_eminus_soup" \
    uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
        --traits e_plus --method lora_combo \
        --combo-spec "ep05_em05=e_plus:0.5,e_minus:0.5;ep05_em025=e_plus:0.5,e_minus:0.25;ep05_em075=e_plus:0.5,e_minus:0.75" \
        "${COMMON_B[@]}"

# ═════════════════════════════════════════════════════════════════════════════
# GROUP C: temp 1.0 spot-check on the LoRA contenders
# Built explicitly (not via COMMON_NEUTRAL) since it overrides temperature.
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

# C1: LoRA scale 0.75 at temp 1.0
run_cell "C1_lora_0.75_t1.0" \
    uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
        --traits e_plus --method lora \
        --scale-points 0.75 \
        "${COMMON_C[@]}"

# C2: LoRA scale 1.00 at temp 1.0
run_cell "C2_lora_1.00_t1.0" \
    uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
        --traits e_plus --method lora \
        --scale-points 1.0 \
        "${COMMON_C[@]}"

# ═════════════════════════════════════════════════════════════════════════════
# GROUP D: E- symmetric mirror at temp 0.7
# Output suffix _t0.7_steering_eminus to keep separate from existing E+ runs.
# ═════════════════════════════════════════════════════════════════════════════
COMMON_D=(
    "${COMMON_NEUTRAL[@]}"
    --output-suffix "_t0.7_steering_eminus"
    --conditions baseline
    --vllm
)

# D1: base on E- pressure scenarios at temp 0.7 (we have temp 1.0 already)
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

# D2: sysprompt-induce-LOW on neutral (mirror of induce-high in cell 4)
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

# D3: E- LoRA scale sweep on neutral
run_cell "D3_eminus_lora_sweep_neutral" \
    uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
        --traits e_minus --method lora \
        --scale-points 0.25,0.5,0.75,1.0 \
        "${COMMON_D[@]}"

# D4: E- actcap sweep on neutral (uses freshly-recomputed e_minus axis)
# NOTE: HF transformers + hooks — slower than vLLM (~30 min/fraction).
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
echo "[$(date)] ===== ALL OVERNIGHT CELLS DONE =====" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
printf "%-46s  %-12s  %s\n" "cell" "status" "elapsed" | tee -a "$MASTER_LOG"
printf "%-46s  %-12s  %s\n" "$(printf '%.0s-' {1..46})" "$(printf '%.0s-' {1..12})" "$(printf '%.0s-' {1..8})" | tee -a "$MASTER_LOG"
for i in "${!CELL_NAMES[@]}"; do
    printf "%-46s  %-12s  %s\n" "${CELL_NAMES[$i]}" "${CELL_STATUS[$i]}" "${CELL_TIMES[$i]}" | tee -a "$MASTER_LOG"
done
echo "" | tee -a "$MASTER_LOG"
echo "Master log: $MASTER_LOG"
