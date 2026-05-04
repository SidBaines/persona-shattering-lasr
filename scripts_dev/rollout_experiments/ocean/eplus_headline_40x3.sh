#!/usr/bin/env bash
# Headline-quality rerun for the 4-method E+ steering figure.
#
# 4 cells × 40 prompts × 3 rollouts × 15 turns. Same neutral psychometric
# pool, deterministic via SEED=42 in generate_rollouts.py. The first 10
# prompts are identical to the 10 we used at 10x2; we're rerunning the
# whole 40-prompt set rather than complicating the merge.
#
# Output suffix: _t0.7_main (separates from existing 10x2 cells).
#
# Cells (chosen contenders from earlier qualitative inspection):
#   M1: base
#   M2: E+ LoRA scale 0.75 (cleaner than 1.00 — less caps-and-cliches register)
#   M4: actcap frac 0.85 (matched-trait-neighbourhood with M2; some bro-register
#       degradation visible but cleaner than frac 1.00 tone-loop mode)
#   M5: sysprompt-induce E+
#
# (Negative-LoRA / user-roleplay are appendix material at 10x2 fidelity, not
# rerun here.)
#
# Estimated total: ~1-1.5h (M4 actcap is HF transformers, ~30 min).
#
# Usage:
#   tmux new -s headline
#   bash scripts_dev/rollout_experiments/ocean/eplus_headline_40x3.sh

set -uo pipefail
cd "$(dirname "$0")/../../.."
git pull --ff-only || echo "WARN: git pull failed - continuing"

# Disk pre-flight (LoRA baking + vLLM cache + scratch outputs)
FREE_GB=$(df -BG --output=avail / | tail -1 | tr -d 'G ')
if [ "${FREE_GB:-0}" -lt 15 ]; then
    echo "ERROR: only ${FREE_GB:-?} GB free on / - need at least 15 GB."
    exit 1
fi
echo "[$(date)] disk pre-flight: ${FREE_GB} GB free OK"

mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="logs/headline_master_${TS}.log"
echo "[$(date)] === eplus headline 40x3 rerun starting ===" | tee -a "$MASTER_LOG"

declare -a CELL_NAMES=()
declare -a CELL_STATUS=()
declare -a CELL_TIMES=()

run_cell() {
    local name="$1"; shift
    local log_file="logs/${name}_${TS}.log"
    echo "" | tee -a "$MASTER_LOG"
    echo "[$(date)] === $name ===" | tee -a "$MASTER_LOG"
    local free=$(df -BG --output=avail / | tail -1 | tr -d 'G ')
    echo "[$(date)] disk: ${free} GB free" | tee -a "$MASTER_LOG"
    if [ "${free:-0}" -lt 5 ]; then
        echo "[$(date)] ABORT: only ${free} GB free, refusing to start cell $name" | tee -a "$MASTER_LOG"
        CELL_NAMES+=("$name"); CELL_STATUS+=("ABORTED(disk)"); CELL_TIMES+=("0s")
        return
    fi
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
    --num-rollouts 3
    --num-turns 15
    --max-samples 40
    --assistant-temperature 0.7
    --user-model openai/gpt-4.1-mini
    --assistant-max-new-tokens 512
    --output-suffix "_t0.7_main"
    --conditions baseline
    --vllm
)

# M1: base
run_cell "M1_base_neutral_40x3" \
    uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
        --traits e_plus --method base \
        "${COMMON[@]}"

# M2: E+ LoRA scale 0.75 (cleaner than 1.00 in qualitative inspection)
run_cell "M2_eplus_lora_0.75_40x3" \
    uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
        --traits e_plus --method lora \
        --scale-points 0.75 \
        "${COMMON[@]}"

# M4: actcap frac 0.85 (matched-trait neighbourhood with M2; no --vllm,
# HF transformers + hooks)
run_cell "M4_eplus_actcap_0.85_40x3" \
    uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
        --traits e_plus --method activation_capping \
        --fractions=0.85 \
        --num-rollouts 3 --num-turns 15 \
        --max-samples 40 \
        --assistant-temperature 0.7 \
        --user-model openai/gpt-4.1-mini \
        --assistant-max-new-tokens 512 \
        --output-suffix "_t0.7_main" \
        --conditions baseline

# M5: sysprompt-induce E+
run_cell "M5_sysprompt_high_40x3" \
    uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
        --traits e_plus --method base \
        --conditions sysprompt_elicit \
        --sysprompt-elicit-directions high \
        --num-rollouts 3 --num-turns 15 \
        --max-samples 40 \
        --assistant-temperature 0.7 \
        --user-model openai/gpt-4.1-mini \
        --assistant-max-new-tokens 512 \
        --output-suffix "_t0.7_main" \
        --vllm

# Summary
echo "" | tee -a "$MASTER_LOG"
echo "[$(date)] ===== HEADLINE RERUN DONE =====" | tee -a "$MASTER_LOG"
printf "%-32s  %-12s  %s\n" "cell" "status" "elapsed" | tee -a "$MASTER_LOG"
for i in "${!CELL_NAMES[@]}"; do
    printf "%-32s  %-12s  %s\n" "${CELL_NAMES[$i]}" "${CELL_STATUS[$i]}" "${CELL_TIMES[$i]}" | tee -a "$MASTER_LOG"
done
echo "Master log: $MASTER_LOG"
