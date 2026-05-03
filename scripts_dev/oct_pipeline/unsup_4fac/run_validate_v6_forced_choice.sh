#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Run the F0 forced-choice validation against:
#   1. baseline (no LoRA)
#   2. v6 paired-DPO amp
#   3. v6 paired-DPO sup
#   4. v6 single-teacher amp
#   5. v6 single-teacher sup
#
# Each run loads the base Llama-3.1-8B-Instruct model fresh in vLLM
# (necessary because adapter swapping mid-run requires extra plumbing
# we haven't wired up yet). All five runs use the same questionnaire and
# the same scoring procedure, so the resulting summary JSONs are
# directly comparable.
#
# Pass the GPU id as the first positional arg (default 0). Skip stages
# via env vars:
#   SKIP_BASELINE=1
#   SKIP_PAIRED_AMP=1 SKIP_PAIRED_SUP=1
#   SKIP_SINGLE_AMP=1 SKIP_SINGLE_SUP=1
#
# Usage:
#   bash scripts_dev/oct_pipeline/unsup_4fac/run_validate_v6_forced_choice.sh <gpu_id>
# ─────────────────────────────────────────────────────────────────────────────
set -o pipefail

GPU="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1

REPO_ROOT="/root/persona-shattering-lasr"
SCRIPT="${REPO_ROOT}/scripts_dev/oct_pipeline/unsup_4fac/validate_lora_forced_choice.py"
LOG_DIR="${REPO_ROOT}/scratch/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

PAIRED_DPO_AMP="persona-shattering-lasr/monorepo::fine_tuning/llama-3.1-8b-it/unsupervised/conviction/amplifier/vunsup_4fac_paired_dpo_v6/lora/conviction_amplifying_v6_unsup_4fac-dpo"
PAIRED_DPO_SUP="persona-shattering-lasr/monorepo::fine_tuning/llama-3.1-8b-it/unsupervised/conviction/suppressor/vunsup_4fac_paired_dpo_v6/lora/conviction_suppressing_v6_unsup_4fac-dpo"
SINGLE_AMP="persona-shattering-lasr/monorepo::fine_tuning/llama-3.1-8b-it/unsupervised/conviction/amplifier/vunsup_4fac_v6/lora/conviction_amplifying_v6_unsup_4fac-dpo"
SINGLE_SUP="persona-shattering-lasr/monorepo::fine_tuning/llama-3.1-8b-it/unsupervised/conviction/suppressor/vunsup_4fac_v6/lora/conviction_suppressing_v6_unsup_4fac-dpo"

FAILED=()

run_one() {
    # $1 = label, $2 = adapter (empty string for baseline)
    local LABEL="$1"
    local ADAPTER="$2"
    local LOG="${LOG_DIR}/${LABEL}_${STAMP}.log"

    echo
    echo "================================================================"
    echo "  ${LABEL}"
    echo "  GPU:     ${GPU}"
    if [ -n "$ADAPTER" ]; then
        echo "  adapter: ${ADAPTER}"
    else
        echo "  adapter: (baseline, no LoRA)"
    fi
    echo "  log:     ${LOG}"
    echo "================================================================"

    local ADAPTER_ARG=()
    if [ -n "$ADAPTER" ]; then
        ADAPTER_ARG=(--adapter "$ADAPTER")
    fi

    if ! stdbuf -oL -eL uv run python "$SCRIPT" \
            --label "$LABEL" \
            "${ADAPTER_ARG[@]}" \
            2>&1 | stdbuf -oL -eL tee "$LOG"; then
        echo "!!! FAILED: ${LABEL}"
        FAILED+=("$LABEL")
    fi
}

if [ "${SKIP_BASELINE:-0}" = "1" ]; then
    echo "skip baseline"
else
    run_one "baseline_fc_v6" ""
fi
if [ "${SKIP_PAIRED_AMP:-0}" = "1" ]; then
    echo "skip paired amp"
else
    run_one "conviction_amp_dpo_v6_fc" "$PAIRED_DPO_AMP"
fi
if [ "${SKIP_PAIRED_SUP:-0}" = "1" ]; then
    echo "skip paired sup"
else
    run_one "conviction_sup_dpo_v6_fc" "$PAIRED_DPO_SUP"
fi
if [ "${SKIP_SINGLE_AMP:-0}" = "1" ]; then
    echo "skip single amp"
else
    run_one "conviction_amp_dpo_v6_fc_singleteacher" "$SINGLE_AMP"
fi
if [ "${SKIP_SINGLE_SUP:-0}" = "1" ]; then
    echo "skip single sup"
else
    run_one "conviction_sup_dpo_v6_fc_singleteacher" "$SINGLE_SUP"
fi

echo
echo "================================================================"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "  All forced-choice runs complete."
else
    echo "  Forced-choice runs with failures:"
    for f in "${FAILED[@]}"; do echo "    - $f"; done
fi
echo
echo "Summaries:"
for L in baseline_fc_v6 \
         conviction_amp_dpo_v6_fc conviction_sup_dpo_v6_fc \
         conviction_amp_dpo_v6_fc_singleteacher conviction_sup_dpo_v6_fc_singleteacher; do
    echo "  scratch/factor_inspect/validate_fc/${L}/${L}_summary.json"
done
echo "================================================================"
