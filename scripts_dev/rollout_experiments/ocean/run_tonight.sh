#!/usr/bin/env bash
# Sequential overnight runner for the E- direction Part 2 cross-method matrix.
#
# Cells (in priority order; HF skip_completed avoids re-running existing ones):
#   1. Validate new v2 scenarios on base (5 new E- scenarios)
#   2. Neutral baseline (no scenario, no sysprompt, no LoRA)
#   3. E+ LoRA scale sweep on v1 winners scenarios (0.25, 0.5, 0.75)
#   4. E+ LoRA scale sweep on sysprompt elicitation (0.25, 0.5, 1.0)
#   5. Activation capping on v1 winners scenarios (with axis-version caveat)
#   6. Activation capping on sysprompt elicitation (same caveat)
#
# Estimated total time on A100 80GB: ~4-5 hours sequential.
# Activation capping cells run on HF transformers (no vLLM), so they're slower.
#
# Usage:
#   tmux new -s tonight
#   bash scripts_dev/rollout_experiments/ocean/run_tonight.sh
#   # Ctrl-b d to detach. Reattach: tmux attach -t tonight
#
# Survives any single-cell failure: set -e is NOT enabled, so subsequent
# cells run even if an earlier one errors. Each cell's exit status is logged.
# Use `tail -f logs/tonight_master_*.log` to watch overall progress.

set -uo pipefail   # strict on undefined vars and pipe failures, but NOT -e
                   # — we want to keep firing cells even if one fails

# ── Project root ─────────────────────────────────────────────────────────────
cd "$(dirname "$0")/../../.."

git pull --ff-only || echo "WARN: git pull failed — continuing with current checkout"

# ── Subset definitions ───────────────────────────────────────────────────────
WINNERS_V1="e_minus_solo_cabin_weekend_01,e_minus_grief_evening_01,e_minus_rainy_afternoon_reading_01,e_minus_astronomy_clear_night_01"
NEW_V2="e_minus_old_letters_drawer_06,e_minus_late_night_regret_07,e_minus_translating_haiku_08,e_minus_dawn_light_question_09,e_minus_estranged_sibling_letter_10"

# ── Setup ────────────────────────────────────────────────────────────────────
mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="logs/tonight_master_${TS}.log"
echo "[$(date)] === tonight.sh starting; master log: $MASTER_LOG ===" | tee -a "$MASTER_LOG"

# Track per-cell results so we get a summary at the end
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

# ── Common flags shared across all cells ─────────────────────────────────────
COMMON_FLAGS=(
    --num-rollouts 3
    --num-turns 15
    --user-model openai/gpt-4.1-mini
    --assistant-max-new-tokens 512
)
VLLM_FLAGS=(--vllm)

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1: Validate v2 scenarios on base (no LoRA, no sysprompt)
# Confirms which new scenarios drift before committing them to LoRA matrix.
# ─────────────────────────────────────────────────────────────────────────────
run_cell "01_v2_scenarios_base" \
    uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
        --traits e_minus --method base \
        --conditions pressure_scenarios \
        --scenario-ids "$NEW_V2" \
        "${COMMON_FLAGS[@]}" "${VLLM_FLAGS[@]}"

# ─────────────────────────────────────────────────────────────────────────────
# CELL 2: Neutral baseline — psychometric prompts, no scenario, no sysprompt
# Reference for "natural drift" with no contextual or weight pressure.
# Same dataset as sysprompt_elicit so seeds match for direct comparison.
# ─────────────────────────────────────────────────────────────────────────────
run_cell "02_neutral_baseline" \
    uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
        --traits e_minus --method base \
        --conditions baseline \
        --max-samples 8 \
        "${COMMON_FLAGS[@]}" "${VLLM_FLAGS[@]}"

# ─────────────────────────────────────────────────────────────────────────────
# CELL 3: E+ LoRA scale sweep on v1 winners scenarios
# Find the "neutralization" sweet spot below scale +1 where extraversion
# stays near 0 (drift prevented) without coherence collapse.
# ─────────────────────────────────────────────────────────────────────────────
run_cell "03_eplus_lora_sweep_winners" \
    uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
        --traits e_plus --method lora \
        --scale-points "0.25,0.5,0.75" \
        --conditions pressure_scenarios \
        --scenario-ids "$WINNERS_V1" \
        "${COMMON_FLAGS[@]}" "${VLLM_FLAGS[@]}"

# ─────────────────────────────────────────────────────────────────────────────
# CELL 4: E+ LoRA scale sweep on sysprompt elicitation
# Tests whether LoRA's coherence collapse on scenarios is also there with
# sysprompt-driven elicitation. Includes scale 1.0 since we don't yet have
# that sysprompt+LoRA cell.
# ─────────────────────────────────────────────────────────────────────────────
run_cell "04_eplus_lora_sysprompt_low" \
    uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
        --traits e_plus --method lora \
        --scale-points "0.25,0.5,1.0" \
        --conditions sysprompt_elicit \
        --sysprompt-elicit-directions low \
        --max-samples 8 \
        "${COMMON_FLAGS[@]}" "${VLLM_FLAGS[@]}"

# ─────────────────────────────────────────────────────────────────────────────
# CELL 5: Activation capping on v1 winners scenarios
# Comparator method to LoRA. Uses the e_plus axis (caveat: axis was trained
# against vanton1, not vanton4_paired_dpo). vLLM is silently ignored for
# activation capping; falls back to HF transformers (slower).
#
# Fractions: 0 = no cap (pin to base mean), positive = push toward E+,
# negative = push toward E-. We sweep mostly positive since the goal is
# "prevent E- drift" in scenarios; a couple negative for symmetry.
# ─────────────────────────────────────────────────────────────────────────────
run_cell "05_actcap_winners" \
    uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
        --traits e_plus --method activation_capping \
        --fractions "-0.5,0.0,0.5,1.0" \
        --conditions pressure_scenarios \
        --scenario-ids "$WINNERS_V1" \
        "${COMMON_FLAGS[@]}"
        # NOTE: no --vllm; activation capping uses HF transformers

# ─────────────────────────────────────────────────────────────────────────────
# CELL 6: Activation capping on sysprompt elicitation
# Same comparator, different elicitation.
# ─────────────────────────────────────────────────────────────────────────────
run_cell "06_actcap_sysprompt_low" \
    uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
        --traits e_plus --method activation_capping \
        --fractions "-0.5,0.0,0.5,1.0" \
        --conditions sysprompt_elicit \
        --sysprompt-elicit-directions low \
        --max-samples 8 \
        "${COMMON_FLAGS[@]}"

# ── Summary ─────────────────────────────────────────────────────────────────
echo "" | tee -a "$MASTER_LOG"
echo "[$(date)] ===== ALL CELLS DONE =====" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
printf "%-32s  %-12s  %s\n" "cell" "status" "elapsed" | tee -a "$MASTER_LOG"
printf "%-32s  %-12s  %s\n" "$(printf '%.0s-' {1..32})" "$(printf '%.0s-' {1..12})" "$(printf '%.0s-' {1..8})" | tee -a "$MASTER_LOG"
for i in "${!CELL_NAMES[@]}"; do
    printf "%-32s  %-12s  %s\n" "${CELL_NAMES[$i]}" "${CELL_STATUS[$i]}" "${CELL_TIMES[$i]}" | tee -a "$MASTER_LOG"
done
echo "" | tee -a "$MASTER_LOG"
echo "Master log: $MASTER_LOG"
