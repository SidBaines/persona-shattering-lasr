#!/usr/bin/env bash
# Minimal O± steering matrix to test whether the LoRA-vs-sysprompt
# floor asymmetry replicates on a different OCEAN trait.
#
# Cells (10 prompts x 2 rollouts x 15 turns - matches our 10x2 baseline):
#   1. recompute o_plus axis (~30 min, prereq for actcap)
#   2. recompute o_minus axis (~30 min, prereq for actcap)
#   3. base on neutral (only one - shared between O+ and O-)
#   4. O+ LoRA scale 0.75
#   5. O+ LoRA scale 1.00
#   6. O+ actcap frac 0.75
#   7. O+ actcap frac 1.00
#   8. O+ sysprompt-induce high
#   9. O- LoRA scale 0.75
#   10. O- LoRA scale 1.00
#   11. O- actcap frac 0.75
#   12. O- actcap frac 1.00
#   13. O- sysprompt-induce low
#
# After axes finish, we need to manually update OCEAN_REGISTRY's
# o_plus and o_minus axis_slug in lora_catalogue.py before the actcap
# cells will work. The script will SKIP actcap cells with a warning if
# axis_slug is still None — they can be rerun later.
#
# Estimated total: ~3-4h (2 axis recomputes + 11 generation cells).
#
# Usage:
#   tmux new -s openness
#   bash scripts_dev/rollout_experiments/ocean/openness_steering_minimal.sh

set -uo pipefail
cd "$(dirname "$0")/../../.."
git pull --ff-only || echo "WARN: git pull failed - continuing"

FREE_GB=$(df -BG --output=avail / | tail -1 | tr -d 'G ')
if [ "${FREE_GB:-0}" -lt 15 ]; then
    echo "ERROR: only ${FREE_GB:-?} GB free on / - need at least 15 GB."
    exit 1
fi

mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="logs/openness_master_${TS}.log"
echo "[$(date)] === O± minimal steering matrix starting ===" | tee -a "$MASTER_LOG"

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

COMMON_NEUTRAL=(
    --num-rollouts 2
    --num-turns 15
    --max-samples 10
    --assistant-temperature 0.7
    --user-model openai/gpt-4.1-mini
    --assistant-max-new-tokens 512
    --output-suffix "_t0.7_steering_o"
    --conditions baseline
)

# ── Phase 1: axis recomputes (prereq for actcap) ──────────────────────────
# These produce o_plus_axis.pt and o_minus_axis.pt at the canonical
# vanton4_paired_dpo location, and the script will warn if you forgot to
# update OCEAN_REGISTRY's axis_slug before running actcap cells.

run_cell "axis_o_plus" \
    uv run python scripts_dev/activation_capping/ocean/paper_versions/compute_axis.py \
        --persona o_plus --force

run_cell "axis_o_minus" \
    uv run python scripts_dev/activation_capping/ocean/paper_versions/compute_axis.py \
        --persona o_minus --force

# ── Manual gate: axis_slug needs to be set in lora_catalogue.py ──────────
# Reading current state:
O_PLUS_AXIS_OK=$(uv run python -c "
from src_dev.common.lora_catalogue import OCEAN_REGISTRY
print('yes' if OCEAN_REGISTRY['o_plus'].axis_slug == 'o_plus' else 'no')
" 2>&1 | tail -1)
O_MINUS_AXIS_OK=$(uv run python -c "
from src_dev.common.lora_catalogue import OCEAN_REGISTRY
print('yes' if OCEAN_REGISTRY['o_minus'].axis_slug == 'o_minus' else 'no')
" 2>&1 | tail -1)

if [ "$O_PLUS_AXIS_OK" != "yes" ]; then
    echo "" | tee -a "$MASTER_LOG"
    echo "[$(date)] WARNING: o_plus axis_slug not set in lora_catalogue.py" | tee -a "$MASTER_LOG"
    echo "  Set OCEAN_REGISTRY['o_plus'].axis_slug = 'o_plus', commit, push," | tee -a "$MASTER_LOG"
    echo "  pull on RunPod, then rerun the O+ actcap cells separately." | tee -a "$MASTER_LOG"
    echo "  (Continuing with non-actcap cells.)" | tee -a "$MASTER_LOG"
fi
if [ "$O_MINUS_AXIS_OK" != "yes" ]; then
    echo "[$(date)] WARNING: o_minus axis_slug not set in lora_catalogue.py" | tee -a "$MASTER_LOG"
fi

# ── Phase 2: generation cells ───────────────────────────────────────────

# Base on neutral (shared)
run_cell "O_base_neutral" \
    uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
        --traits o_plus --method base \
        --vllm \
        "${COMMON_NEUTRAL[@]}"

# O+ LoRA sweep {0.75, 1.0}
run_cell "O_oplus_lora_sweep" \
    uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
        --traits o_plus --method lora \
        --scale-points 0.75,1.0 \
        --vllm \
        "${COMMON_NEUTRAL[@]}"

# O+ sysprompt induce high
run_cell "O_oplus_sysprompt_high" \
    uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
        --traits o_plus --method base \
        --conditions sysprompt_elicit \
        --sysprompt-elicit-directions high \
        --num-rollouts 2 --num-turns 15 \
        --max-samples 10 \
        --assistant-temperature 0.7 \
        --user-model openai/gpt-4.1-mini \
        --assistant-max-new-tokens 512 \
        --output-suffix "_t0.7_steering_o" \
        --vllm

# O- LoRA sweep {0.75, 1.0}
run_cell "O_ominus_lora_sweep" \
    uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
        --traits o_minus --method lora \
        --scale-points 0.75,1.0 \
        --vllm \
        "${COMMON_NEUTRAL[@]}"

# O- sysprompt induce low
run_cell "O_ominus_sysprompt_low" \
    uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
        --traits o_minus --method base \
        --conditions sysprompt_elicit \
        --sysprompt-elicit-directions low \
        --num-rollouts 2 --num-turns 15 \
        --max-samples 10 \
        --assistant-temperature 0.7 \
        --user-model openai/gpt-4.1-mini \
        --assistant-max-new-tokens 512 \
        --output-suffix "_t0.7_steering_o" \
        --vllm

# O+/O- actcap (skipped if axis_slug not set; can be rerun separately)
if [ "$O_PLUS_AXIS_OK" = "yes" ]; then
    run_cell "O_oplus_actcap_sweep" \
        uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
            --traits o_plus --method activation_capping \
            --fractions=0.75,1.0 \
            "${COMMON_NEUTRAL[@]}"
fi
if [ "$O_MINUS_AXIS_OK" = "yes" ]; then
    run_cell "O_ominus_actcap_sweep" \
        uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
            --traits o_minus --method activation_capping \
            --fractions=0.75,1.0 \
            "${COMMON_NEUTRAL[@]}"
fi

# Summary
echo "" | tee -a "$MASTER_LOG"
echo "[$(date)] ===== O± MATRIX DONE =====" | tee -a "$MASTER_LOG"
printf "%-32s  %-12s  %s\n" "cell" "status" "elapsed" | tee -a "$MASTER_LOG"
for i in "${!CELL_NAMES[@]}"; do
    printf "%-32s  %-12s  %s\n" "${CELL_NAMES[$i]}" "${CELL_STATUS[$i]}" "${CELL_TIMES[$i]}" | tee -a "$MASTER_LOG"
done
echo "Master log: $MASTER_LOG"
echo ""
echo "If O+/O- actcap cells were skipped, set axis_slug in lora_catalogue.py,"
echo "commit, push, pull on RunPod, then rerun those two cells."
