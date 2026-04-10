#!/usr/bin/env bash
# Run trait logprob sweeps for all LoRA adapters.
#
# Re-entrant: each completed eval creates a .done sentinel.  If the script is
# interrupted and restarted, already-done items are skipped.  To force a full
# re-run, delete the LOG_DIR.
#
# Usage:
#   bash scripts_dev/personality_evals/run_all_trait_logprobs.sh
set -euo pipefail

LOG_DIR="scratch/logs/trait_logprobs_batch"
mkdir -p "$LOG_DIR"
CONFIG_MODULE="scripts_dev.personality_evals.configs.ocean.trait.run_adapter"

run_eval() {
    local key_type="$1" key_value="$2"
    local log_file="$LOG_DIR/${key_value}.log"
    local done_file="$LOG_DIR/${key_value}.done"

    if [[ -f "$done_file" ]]; then
        echo "[$(date '+%H:%M:%S')] SKIP (already done): $key_value"
        return 0
    fi

    echo "[$(date '+%H:%M:%S')] Running: $key_value"
    if env "${key_type}=${key_value}" uv run python -m src_dev.evals suite \
            --config-module "$CONFIG_MODULE" 2>&1 | tee "$log_file"; then
        touch "$done_file"
        echo "[$(date '+%H:%M:%S')] DONE: $key_value"
    else
        echo "[$(date '+%H:%M:%S')] FAILED: $key_value (see $log_file)"
    fi
}

# ── Step 0: Baseline (computed once, reused by all sweeps) ──────────────
run_eval ADAPTER_KEY __baseline__

# ── Step 1: Priority — vanton1 standardised set (amp + sup per trait) ───
for a in \
    a_plus_vanton1  a_minus_vanton1 \
    c_plus_vanton1  c_minus_vanton1 \
    e_plus_vanton1  e_minus_vanton1 \
    n_plus_vanton1  n_minus_vanton1 \
    o_plus_vanton1  o_minus_vanton1 \
; do
    run_eval ADAPTER_KEY "$a"
done

# ── Step 2: Other adapter versions ──────────────────────────────────────
for a in \
    e_plus_v3 \
    n_plus_v4 \
    a_plus_vanton2 \
    a_plus_v1       a_minus_v2 \
    c_plus_v1_souped c_minus_v2 \
    e_plus_v1       e_plus_v2 \
    control_diff_words control_empty_traits \
; do
    run_eval ADAPTER_KEY "$a"
done

# ── Step 3: Combinations (add after inspecting sweep results) ───────────
# Uncomment and populate after reviewing individual sweep results:
# for c in \
#     a_plus_minus_vanton1 \
#     c_plus_minus_vanton1 \
#     e_plus_minus_vanton1 \
#     n_plus_minus_vanton1 \
#     o_plus_minus_vanton1 \
# ; do
#     run_eval COMBO_KEY "$c"
# done

echo ""
echo "[$(date '+%H:%M:%S')] All done."
echo "Results: scratch/evals/ocean/trait/"
echo "Logs:    $LOG_DIR/"
