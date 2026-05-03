#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Persona-mediated forced-choice F0 validation runner. Companion to
# run_validate_v6_forced_choice.sh (which is direct, no-persona).
#
# Administers the f0_forced_choice_v1 questionnaire to the same 200-persona
# subsample that validate_lora.py uses for the FA-based F0 scoring, so we
# can correlate per-persona FC scores with per-persona FA F0 scores
# (validates that the FC questionnaire is measuring the engaged-agency
# construct cleanly, before we trust its per-LoRA shifts).
#
# Each LoRA is trained-and-validated independently (validate_lora.py needs
# to have already run for the same LoRA so its baseline_scores file is
# available for FC↔FA correlation).
#
# Pre-reqs:
#   * scratch/factor_inspect/validate/{conviction_amp,sup}_dpo_v6/<>_scores.npz
#     must exist (output of validate_lora.py for the v6 paired-DPO and/or
#     single-teacher runs). The persona-mediated FC script reads
#     baseline_scores / lora_scores from these to compute correlation.
#
# Skip stages via env vars:
#   SKIP_BASELINE=1 SKIP_PAIRED_AMP=1 SKIP_PAIRED_SUP=1
#   SKIP_SINGLE_AMP=1 SKIP_SINGLE_SUP=1
#
# Usage:
#   bash scripts_dev/oct_pipeline/unsup_4fac/run_validate_v6_fc_persona.sh <gpu_id>
# ─────────────────────────────────────────────────────────────────────────────
set -o pipefail

GPU="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1

REPO_ROOT="/root/persona-shattering-lasr"
SCRIPT="${REPO_ROOT}/scripts_dev/oct_pipeline/unsup_4fac/validate_lora_fc_persona.py"
LOG_DIR="${REPO_ROOT}/scratch/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

N_PERSONAS="${N_PERSONAS:-200}"

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
    echo "  ${LABEL}  (persona-mediated FC)"
    echo "  GPU:        ${GPU}"
    echo "  n_personas: ${N_PERSONAS}"
    if [ -n "$ADAPTER" ]; then
        echo "  adapter:    ${ADAPTER}"
    else
        echo "  adapter:    (baseline, no LoRA)"
    fi
    echo "  log:        ${LOG}"
    echo "================================================================"

    local ADAPTER_ARG=()
    if [ -n "$ADAPTER" ]; then
        ADAPTER_ARG=(--adapter "$ADAPTER")
    fi

    if ! stdbuf -oL -eL uv run python "$SCRIPT" \
            --label "$LABEL" \
            --n-personas "$N_PERSONAS" \
            "${ADAPTER_ARG[@]}" \
            2>&1 | stdbuf -oL -eL tee "$LOG"; then
        echo "!!! FAILED: ${LABEL}"
        FAILED+=("$LABEL")
    fi
}

if [ "${SKIP_BASELINE:-0}" = "1" ]; then
    echo "skip baseline"
else
    run_one "baseline_fc_persona" ""
fi
if [ "${SKIP_PAIRED_AMP:-0}" = "1" ]; then
    echo "skip paired amp"
else
    run_one "conviction_amp_dpo_v6_fc_persona" "$PAIRED_DPO_AMP"
fi
if [ "${SKIP_PAIRED_SUP:-0}" = "1" ]; then
    echo "skip paired sup"
else
    run_one "conviction_sup_dpo_v6_fc_persona" "$PAIRED_DPO_SUP"
fi
if [ "${SKIP_SINGLE_AMP:-0}" = "1" ]; then
    echo "skip single amp"
else
    run_one "conviction_amp_dpo_v6_fc_persona_singleteacher" "$SINGLE_AMP"
fi
if [ "${SKIP_SINGLE_SUP:-0}" = "1" ]; then
    echo "skip single sup"
else
    run_one "conviction_sup_dpo_v6_fc_persona_singleteacher" "$SINGLE_SUP"
fi

echo
echo "================================================================"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "  All persona-mediated FC runs complete."
else
    echo "  Persona-mediated FC runs with failures:"
    for f in "${FAILED[@]}"; do echo "    - $f"; done
fi
echo
echo "Per-persona scores + summaries:"
for L in baseline_fc_persona \
         conviction_amp_dpo_v6_fc_persona conviction_sup_dpo_v6_fc_persona \
         conviction_amp_dpo_v6_fc_persona_singleteacher conviction_sup_dpo_v6_fc_persona_singleteacher; do
    echo "  scratch/factor_inspect/validate_fc_persona/${L}/${L}_summary.json"
    echo "  scratch/factor_inspect/validate_fc_persona/${L}/${L}_scores.npz"
done
echo "================================================================"
