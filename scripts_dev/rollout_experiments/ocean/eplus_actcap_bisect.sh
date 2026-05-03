#!/usr/bin/env bash
# Binary search for the actcap "sweet spot" fraction in (0.75, 1.0).
#
# Context: on neutral prompts at temp 0.7, actcap frac 0.75 lifts
# extraversion to +1.55 (coh 8.17) while frac 1.00 reaches +3.12 (coh 6.24).
# That's a large gap with a coherence cliff — worth bisecting.
#
# This script runs ONE fraction at a time. After it lands, decide where
# to bisect next based on observed trait+coherence values, then update
# FRACTIONS below and rerun.
#
# Pass --frac VALUE on the CLI to override (e.g. `bash ... --frac 0.825`).
#
# Usage:
#   tmux new -s actcap_bisect
#   bash scripts_dev/rollout_experiments/ocean/eplus_actcap_bisect.sh         # frac 0.85
#   bash scripts_dev/rollout_experiments/ocean/eplus_actcap_bisect.sh --frac 0.925
#
# Output suffix matches eplus_steering_runs.sh (_t0.7_steering) so the new
# variant lands alongside frac_0.25..1.00 under the same sweep dir.

set -uo pipefail

cd "$(dirname "$0")/../../.."

FRAC="0.85"
if [[ "${1:-}" == "--frac" ]]; then
    FRAC="${2:?--frac requires a value}"
fi

git pull --ff-only || echo "WARN: git pull failed — continuing with current checkout"

mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/actcap_bisect_frac${FRAC}_${TS}.log"

echo "[$(date)] === actcap bisect: frac=$FRAC ===" | tee -a "$LOG"
echo "[$(date)] log: $LOG" | tee -a "$LOG"

uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
    --traits e_plus --method activation_capping \
    --fractions="$FRAC" \
    --conditions baseline \
    --max-samples 10 \
    --num-rollouts 2 \
    --num-turns 15 \
    --assistant-temperature 0.7 \
    --user-model openai/gpt-4.1-mini \
    --assistant-max-new-tokens 512 \
    --output-suffix "_t0.7_steering" \
    2>&1 | tee -a "$LOG"

EXIT=${PIPESTATUS[0]}
if [ "$EXIT" -eq 0 ]; then
    echo "[$(date)] === DONE frac=$FRAC ===" | tee -a "$LOG"
    echo "  Will land at: rollout_sweep_activation_capping_t0.7_steering/frac_$FRAC/baseline/" | tee -a "$LOG"
else
    echo "[$(date)] === FAILED exit=$EXIT ===" | tee -a "$LOG"
    exit "$EXIT"
fi
