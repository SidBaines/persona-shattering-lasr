#!/usr/bin/env bash
# E+ steering vibes-check matrix.
#
# Question: how do different methods (LoRA, activation capping, sysprompt,
# user-roleplay) compare for INDUCING the E+ persona on a model whose default
# is mildly E-? This is the Part 1 / steering question (vs Part 2's drift
# prevention). We're not picking winners — just generating data to plot all
# methods on the same axes and see qualitative shape differences.
#
# Matrix:
#   1. Base on neutral prompts                       — no intervention floor
#   2. LoRA E+ scale sweep on neutral prompts        — pure weight steering
#   3. Actcap E+ fraction sweep on neutral prompts   — pure activation steering
#   4. Sysprompt-induce E+ on neutral prompts        — prompt-only steering
#   5. Base on E+ pressure scenarios                 — user-roleplay steering
#
# Counts:
#   Neutral prompts: 10 (matched to E+ scenario count)
#   E+ scenarios:    10 (all of them, both v1 and v2)
#   Rollouts/prompt: 2
#   Turns:           15
#   Sweep points:    {0.25, 0.50, 0.75, 1.00}  (skip 0.0 — that's the base cell)
#
# Temperature: 0.7. (Default has been 1.0; lower temp may reduce token-soup
# tail collapse on long outputs. We re-run everything here at 0.7 for
# internal comparison; can re-run cells of interest at 1.0 later.)
#
# Output suffix: "_t0.7_steering" appended to all eval_names so this matrix
# doesn't collide with previous runs on HF.
#
# Estimated total: ~2-3h on A100/H100 (LoRA via vLLM is fast; actcap via
# HF transformers is the slow part).
#
# Usage:
#   tmux new -s steering
#   bash scripts_dev/rollout_experiments/ocean/eplus_steering_runs.sh

set -uo pipefail

cd "$(dirname "$0")/../../.."

git pull --ff-only || echo "WARN: git pull failed — continuing with current checkout"

mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="logs/eplus_steering_master_${TS}.log"
echo "[$(date)] === eplus_steering matrix starting; master log: $MASTER_LOG ===" | tee -a "$MASTER_LOG"

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

# All 10 E+ scenarios from extraversion_pressure_v1.json
EPLUS_SCENARIOS="e_plus_birthday_surprise_01,e_plus_dating_app_match_01,e_plus_team_pitch_hype_01,e_plus_road_trip_companion_01,e_plus_group_vacation_planning_01,e_plus_live_radio_show_warmup_02,e_plus_dj_set_emergency_03,e_plus_improv_warmup_04,e_plus_food_truck_rush_05,e_plus_open_mic_co_writing_06"

# Common flags shared by every cell. Temperature 0.7 across the board.
COMMON_FLAGS=(
    --num-rollouts 2
    --num-turns 15
    --assistant-temperature 0.7
    --user-model openai/gpt-4.1-mini
    --assistant-max-new-tokens 512
    --output-suffix "_t0.7_steering"
)

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1: Base on neutral psychometric prompts (no intervention floor)
# Uses the default --dataset (datasets/psychometric_seed_prompts/v1xAA.jsonl).
# 10 prompts to match the E+ scenario count.
# ─────────────────────────────────────────────────────────────────────────────
run_cell "01_base_neutral_t0.7" \
    uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
        --traits e_plus --method base \
        --conditions baseline \
        --max-samples 10 \
        --vllm \
        "${COMMON_FLAGS[@]}"

# ─────────────────────────────────────────────────────────────────────────────
# CELL 2: LoRA E+ scale sweep on neutral prompts (pure weight steering)
# Skip 0.0 (= base, covered by cell 1). Sweep {0.25, 0.5, 0.75, 1.0}.
# ─────────────────────────────────────────────────────────────────────────────
run_cell "02_lora_neutral_t0.7" \
    uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
        --traits e_plus --method lora \
        --scale-points 0.25,0.5,0.75,1.0 \
        --conditions baseline \
        --max-samples 10 \
        --vllm \
        "${COMMON_FLAGS[@]}"

# ─────────────────────────────────────────────────────────────────────────────
# CELL 3: Actcap E+ fraction sweep on neutral prompts (pure activation steering)
# HF transformers + hooks (slower than vLLM). Skip 0.0.
# ─────────────────────────────────────────────────────────────────────────────
run_cell "03_actcap_neutral_t0.7" \
    uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
        --traits e_plus --method activation_capping \
        --fractions=0.25,0.5,0.75,1.0 \
        --conditions baseline \
        --max-samples 10 \
        "${COMMON_FLAGS[@]}"

# ─────────────────────────────────────────────────────────────────────────────
# CELL 4: Sysprompt-induce E+ on neutral prompts (prompt-only steering)
# No LoRA. Just the canonical OCEAN_DEFINITION-based "be extraverted"
# instruction at the assistant system prompt level.
# ─────────────────────────────────────────────────────────────────────────────
run_cell "04_sysprompt_high_neutral_t0.7" \
    uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
        --traits e_plus --method base \
        --conditions sysprompt_elicit \
        --sysprompt-elicit-directions high \
        --max-samples 10 \
        --vllm \
        "${COMMON_FLAGS[@]}"

# ─────────────────────────────────────────────────────────────────────────────
# CELL 5: Base on all 10 E+ pressure scenarios (user-roleplay steering)
# No LoRA, no actcap, no sysprompt. Just the user-sim playing E+ scenarios
# against the base model.
# ─────────────────────────────────────────────────────────────────────────────
run_cell "05_base_eplus_scenarios_t0.7" \
    uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
        --traits e_plus --method base \
        --conditions pressure_scenarios \
        --scenario-ids "$EPLUS_SCENARIOS" \
        --vllm \
        "${COMMON_FLAGS[@]}"

# ── Summary ─────────────────────────────────────────────────────────────────
echo "" | tee -a "$MASTER_LOG"
echo "[$(date)] ===== ALL STEERING CELLS DONE =====" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
printf "%-40s  %-12s  %s\n" "cell" "status" "elapsed" | tee -a "$MASTER_LOG"
printf "%-40s  %-12s  %s\n" "$(printf '%.0s-' {1..40})" "$(printf '%.0s-' {1..12})" "$(printf '%.0s-' {1..8})" | tee -a "$MASTER_LOG"
for i in "${!CELL_NAMES[@]}"; do
    printf "%-40s  %-12s  %s\n" "${CELL_NAMES[$i]}" "${CELL_STATUS[$i]}" "${CELL_TIMES[$i]}" | tee -a "$MASTER_LOG"
done
echo "" | tee -a "$MASTER_LOG"
echo "Master log: $MASTER_LOG"
