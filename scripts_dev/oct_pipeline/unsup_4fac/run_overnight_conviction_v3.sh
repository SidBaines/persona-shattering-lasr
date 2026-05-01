#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Overnight pipeline for the unsup_4fac F0 (Conviction) v3 retrain.
#
# Sequence, single GPU:
#   1. Generate/check v3 clement-style conviction constitutions.
#   2. Teacher distillation for amplifier and suppressor with K teacher samples.
#   3. Seed paired-DPO data from the two teacher distillation runs.
#   4. Train amplifier and suppressor LoRAs.
#   5. Validate both trained persona LoRAs against the FA questionnaire.
#
# Usage:
#   bash scripts_dev/oct_pipeline/unsup_4fac/run_overnight_conviction_v3.sh <gpu_id>
#
# Suggested tmux launch:
#   mkdir -p scratch/logs
#   LOG=scratch/logs/overnight_conviction_v3_$(date -u +%Y%m%dT%H%M%SZ).log
#   tmux new -d -s conviction_v3 "cd /root/persona-shattering-lasr && bash scripts_dev/oct_pipeline/unsup_4fac/run_overnight_conviction_v3.sh 0 2>&1 | tee $LOG"
#
# Skip phases via env var (set to 1 to skip):
#   SKIP_GENERATE=1   skip constitution generation/check
#   SKIP_DISTILL=1    skip teacher distillation
#   SKIP_SEED=1       skip paired-DPO seeding
#   SKIP_TRAIN=1      skip training
#   SKIP_VALIDATE=1   skip post-train validation
# ─────────────────────────────────────────────────────────────────────────────
set -o pipefail

GPU="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
export MASTER_PORT="$((29500 + GPU))"
export PYTHONUNBUFFERED=1

TRAIT="conviction"
SOURCE_VERSION="${SOURCE_VERSION:-unsup_4fac_v3}"
DEST_VERSION="${DEST_VERSION:-unsup_4fac_paired_dpo_v3}"
TEACHER_K="${TEACHER_K:-3}"
AMP_PAIRING="${AMP_PAIRING:-all}"
N_PERSONAS="${N_PERSONAS:-200}"

CONST_STEM_AMP="conviction_amplifying_v3_unsup_4fac"
CONST_STEM_SUP="conviction_suppressing_v3_unsup_4fac"

REPO_ROOT="/root/persona-shattering-lasr"
LOG_DIR="${REPO_ROOT}/scratch/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OVERALL_LOG="${LOG_DIR}/overnight_conviction_v3_${STAMP}.log"

phase_header() {
    echo
    echo "################################################################"
    echo "# $1"
    echo "# $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "################################################################"
}

if [ "${SKIP_GENERATE:-0}" = "1" ]; then
    phase_header "Step 1 SKIPPED: generate/check v3 constitutions"
else
    phase_header "Step 1: generate/check v3 constitutions"
    if ! uv run python "${REPO_ROOT}/scripts_dev/oct_pipeline/unsup_4fac/generate_v3_conviction_constitutions.py"; then
        echo "!!! Step 1 FAILED."
        exit 1
    fi
fi

if [ "${SKIP_DISTILL:-0}" = "1" ]; then
    phase_header "Step 2 SKIPPED: distillation"
else
    phase_header "Step 2: distillation (K=${TEACHER_K}, v=${SOURCE_VERSION})"
    if ! VERSION="$SOURCE_VERSION" TEACHER_K="$TEACHER_K" \
            CONST_STEM_AMP="$CONST_STEM_AMP" CONST_STEM_SUP="$CONST_STEM_SUP" \
            CONCAT_ALL_TRAITS=1 \
            bash "${REPO_ROOT}/scripts_dev/oct_pipeline/unsup_4fac/prep_unsup_4fac_distillation.sh" \
            "$GPU" "$TRAIT"; then
        echo "!!! Step 2 FAILED."
        exit 1
    fi
fi

if [ "${SKIP_SEED:-0}" = "1" ]; then
    phase_header "Step 3 SKIPPED: seed paired-DPO"
else
    phase_header "Step 3: seed paired-DPO (amp_pairing=${AMP_PAIRING}, src=v${SOURCE_VERSION}, dest=v${DEST_VERSION})"
    if ! SOURCE_VERSION="$SOURCE_VERSION" DEST_VERSION="$DEST_VERSION" AMP_PAIRING="$AMP_PAIRING" \
            CONST_STEM_AMP="$CONST_STEM_AMP" CONST_STEM_SUP="$CONST_STEM_SUP" \
            bash "${REPO_ROOT}/scripts_dev/oct_pipeline/unsup_4fac/seed_unsup_4fac_paired_dpo.sh" "$TRAIT"; then
        echo "!!! Step 3 FAILED."
        exit 1
    fi
fi

if [ "${SKIP_TRAIN:-0}" = "1" ]; then
    phase_header "Step 4 SKIPPED: training"
else
    phase_header "Step 4: train paired-DPO persona LoRAs (v=${DEST_VERSION})"
    if ! VERSION="$DEST_VERSION" \
            CONST_STEM_AMP="$CONST_STEM_AMP" CONST_STEM_SUP="$CONST_STEM_SUP" \
            INTRO_CONST_STEM_AMP="$CONST_STEM_AMP" INTRO_CONST_STEM_SUP="$CONST_STEM_SUP" \
            CONCAT_ALL_TRAITS=1 \
            bash "${REPO_ROOT}/scripts_dev/oct_pipeline/unsup_4fac/run_unsup_4fac_paired_dpo.sh" \
            "$GPU" "$TRAIT"; then
        echo "!!! Step 4 FAILED."
        exit 1
    fi
fi

if [ "${SKIP_VALIDATE:-0}" = "1" ]; then
    phase_header "Step 5 SKIPPED: post-train validation"
else
    phase_header "Step 5: post-train validation of v3 amp + sup persona LoRAs"

    AMP_ADAPTER="persona-shattering-lasr/monorepo::fine_tuning/llama-3.1-8b-it/unsupervised/${TRAIT}/amplifier/v${DEST_VERSION}/lora/${CONST_STEM_AMP}-persona"
    SUP_ADAPTER="persona-shattering-lasr/monorepo::fine_tuning/llama-3.1-8b-it/unsupervised/${TRAIT}/suppressor/v${DEST_VERSION}/lora/${CONST_STEM_SUP}-persona"

    AMP_LABEL="${TRAIT}_amp_v3"
    SUP_LABEL="${TRAIT}_sup_v3"
    AMP_VAL_LOG="${LOG_DIR}/${AMP_LABEL}_validate_${STAMP}.log"
    SUP_VAL_LOG="${LOG_DIR}/${SUP_LABEL}_validate_${STAMP}.log"

    VAL_FAILED=()

    echo
    echo "  validating ${AMP_LABEL}"
    echo "    adapter: ${AMP_ADAPTER}"
    echo "    log:     ${AMP_VAL_LOG}"
    if ! stdbuf -oL -eL uv run python "${REPO_ROOT}/scripts_dev/oct_pipeline/unsup_4fac/validate_lora.py" \
            --target "$TRAIT" \
            --adapter "$AMP_ADAPTER" \
            --n-personas "$N_PERSONAS" \
            --label "$AMP_LABEL" \
            2>&1 | stdbuf -oL -eL tee "$AMP_VAL_LOG"; then
        echo "!!! ${AMP_LABEL} validation FAILED"
        VAL_FAILED+=("$AMP_LABEL")
    fi

    echo
    echo "  validating ${SUP_LABEL}"
    echo "    adapter: ${SUP_ADAPTER}"
    echo "    log:     ${SUP_VAL_LOG}"
    if ! stdbuf -oL -eL uv run python "${REPO_ROOT}/scripts_dev/oct_pipeline/unsup_4fac/validate_lora.py" \
            --target "$TRAIT" \
            --adapter "$SUP_ADAPTER" \
            --n-personas "$N_PERSONAS" \
            --label "$SUP_LABEL" \
            2>&1 | stdbuf -oL -eL tee "$SUP_VAL_LOG"; then
        echo "!!! ${SUP_LABEL} validation FAILED"
        VAL_FAILED+=("$SUP_LABEL")
    fi

    if [ ${#VAL_FAILED[@]} -gt 0 ]; then
        echo "!!! Validation failures: ${VAL_FAILED[*]}"
        exit 1
    fi
fi

phase_header "All conviction v3 steps complete."
echo
echo "Summaries:"
echo "  v3 persona AMP:    scratch/factor_inspect/validate/${TRAIT}_amp_v3/${TRAIT}_amp_v3_summary.json"
echo "  v3 persona SUP:    scratch/factor_inspect/validate/${TRAIT}_sup_v3/${TRAIT}_sup_v3_summary.json"
echo
echo "Trained v3 adapters on monorepo:"
echo "  fine_tuning/llama-3.1-8b-it/unsupervised/${TRAIT}/{amplifier,suppressor}/v${DEST_VERSION}/lora/"
echo
echo "Overall log: ${OVERALL_LOG}"
