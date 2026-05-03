#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Overnight pipeline for the unsup_4fac F0 (Conviction) v5 retrain.
#
# v5 = constitution rewrite (7 facets, tag-style trait sentences,
# F3-contaminated facets dropped, validation contamination cleared)
# combined with the OCEAN-vanton4 distillation/DPO recipe:
#   * teacher-k = 1   (no --teacher-k flag)
#   * amp_pairing = first
#   * NO --concat-all-traits-system-prompt
#
# Sequence, single GPU — interleaved per pole so we can read amp results
# while sup is still training:
#   1. Generate/check v5 conviction constitutions.
#   2. Teacher distillation for amplifier and suppressor (K=1, no concat).
#   3. Seed paired-DPO data (amp_pairing=first).
#   4a. Train amplifier LoRA.
#   5a. Validate amplifier persona LoRA.
#   4b. Train suppressor LoRA.
#   5b. Validate suppressor persona LoRA.
#
# Usage:
#   bash scripts_dev/oct_pipeline/unsup_4fac/run_overnight_conviction_v5.sh <gpu_id>
#
# Suggested tmux launch:
#   mkdir -p scratch/logs
#   LOG=scratch/logs/overnight_conviction_v5_$(date -u +%Y%m%dT%H%M%SZ).log
#   tmux new -d -s conviction_v5 "cd /root/persona-shattering-lasr && bash scripts_dev/oct_pipeline/unsup_4fac/run_overnight_conviction_v5.sh 0 2>&1 | tee $LOG"
#
# Skip phases via env var (set to 1 to skip):
#   SKIP_GENERATE=1   skip constitution generation/check
#   SKIP_DISTILL=1    skip teacher distillation
#   SKIP_SEED=1       skip paired-DPO seeding
#   SKIP_TRAIN_AMP=1  skip amp training
#   SKIP_VAL_AMP=1    skip amp validation
#   SKIP_TRAIN_SUP=1  skip sup training
#   SKIP_VAL_SUP=1    skip sup validation
# ─────────────────────────────────────────────────────────────────────────────
set -o pipefail

GPU="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
export MASTER_PORT="$((29500 + GPU))"
export PYTHONUNBUFFERED=1

TRAIT="conviction"
SOURCE_VERSION="${SOURCE_VERSION:-unsup_4fac_v5}"
DEST_VERSION="${DEST_VERSION:-unsup_4fac_paired_dpo_v5}"
TEACHER_K="${TEACHER_K:-}"          # empty ⇒ k=1 (omits --teacher-k flag)
AMP_PAIRING="${AMP_PAIRING:-first}" # 1 pair per prompt, like vanton4
N_PERSONAS="${N_PERSONAS:-200}"

# v5 constitutions: 7 facets, tag-style trait sentences, F3-contam dropped.
CONST_STEM_AMP="conviction_amplifying_v5_unsup_4fac"
CONST_STEM_SUP="conviction_suppressing_v5_unsup_4fac"

REPO_ROOT="/root/persona-shattering-lasr"
LOG_DIR="${REPO_ROOT}/scratch/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OVERALL_LOG="${LOG_DIR}/overnight_conviction_v5_${STAMP}.log"

phase_header() {
    echo
    echo "################################################################"
    echo "# $1"
    echo "# $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "################################################################"
}

train_one_direction() {
    # $1: "amplifier" or "suppressor"
    local DIR="$1"
    DIRECTIONS_TO_RUN="$DIR" \
        VERSION="$DEST_VERSION" \
        CONST_STEM_AMP="$CONST_STEM_AMP" CONST_STEM_SUP="$CONST_STEM_SUP" \
        INTRO_CONST_STEM_AMP="$CONST_STEM_AMP" INTRO_CONST_STEM_SUP="$CONST_STEM_SUP" \
        CONCAT_ALL_TRAITS=0 \
        bash "${REPO_ROOT}/scripts_dev/oct_pipeline/unsup_4fac/run_unsup_4fac_paired_dpo.sh" \
        "$GPU" "$TRAIT"
}

validate_one_direction() {
    # $1: "amplifier" or "suppressor"; $2: const stem; $3: short label suffix
    local DIR="$1"
    local STEM="$2"
    local SUFFIX="$3"
    local ADAPTER="persona-shattering-lasr/monorepo::fine_tuning/llama-3.1-8b-it/unsupervised/${TRAIT}/${DIR}/v${DEST_VERSION}/lora/${STEM}-persona"
    local LABEL="${TRAIT}_${SUFFIX}_v5"
    local VAL_LOG="${LOG_DIR}/${LABEL}_validate_${STAMP}.log"

    echo
    echo "  validating ${LABEL}"
    echo "    adapter: ${ADAPTER}"
    echo "    log:     ${VAL_LOG}"
    stdbuf -oL -eL uv run python "${REPO_ROOT}/scripts_dev/oct_pipeline/unsup_4fac/validate_lora.py" \
        --target "$TRAIT" \
        --adapter "$ADAPTER" \
        --n-personas "$N_PERSONAS" \
        --label "$LABEL" \
        2>&1 | stdbuf -oL -eL tee "$VAL_LOG"
}

# Step 1: constitutions
if [ "${SKIP_GENERATE:-0}" = "1" ]; then
    phase_header "Step 1 SKIPPED: generate/check v5 constitutions"
else
    phase_header "Step 1: generate/check v5 constitutions"
    if ! uv run python "${REPO_ROOT}/scripts_dev/oct_pipeline/unsup_4fac/generate_v5_conviction_constitutions.py"; then
        echo "!!! Step 1 FAILED."
        exit 1
    fi
fi

# Step 2: distillation (both poles, batched — needed before either DPO can run)
if [ "${SKIP_DISTILL:-0}" = "1" ]; then
    phase_header "Step 2 SKIPPED: distillation"
else
    phase_header "Step 2: distillation (K=${TEACHER_K:-1}, v=${SOURCE_VERSION}, no concat-all-traits)"
    if ! VERSION="$SOURCE_VERSION" TEACHER_K="$TEACHER_K" \
            CONST_STEM_AMP="$CONST_STEM_AMP" CONST_STEM_SUP="$CONST_STEM_SUP" \
            CONCAT_ALL_TRAITS=0 \
            bash "${REPO_ROOT}/scripts_dev/oct_pipeline/unsup_4fac/prep_unsup_4fac_distillation.sh" \
            "$GPU" "$TRAIT"; then
        echo "!!! Step 2 FAILED."
        exit 1
    fi
fi

# Step 3: paired-DPO seed (both poles, CPU-only)
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

VAL_FAILED=()

# Step 4a: train amplifier
if [ "${SKIP_TRAIN_AMP:-0}" = "1" ]; then
    phase_header "Step 4a SKIPPED: train amplifier"
else
    phase_header "Step 4a: train amplifier paired-DPO LoRA (v=${DEST_VERSION})"
    if ! train_one_direction amplifier; then
        echo "!!! Step 4a (amp train) FAILED."
        exit 1
    fi
fi

# Step 5a: validate amplifier
if [ "${SKIP_VAL_AMP:-0}" = "1" ]; then
    phase_header "Step 5a SKIPPED: validate amplifier"
else
    phase_header "Step 5a: validate amplifier persona LoRA (n=${N_PERSONAS})"
    if ! validate_one_direction amplifier "$CONST_STEM_AMP" amp; then
        echo "!!! Step 5a (amp validate) FAILED — continuing to sup."
        VAL_FAILED+=("conviction_amp_v5")
    fi
fi

# Step 4b: train suppressor
if [ "${SKIP_TRAIN_SUP:-0}" = "1" ]; then
    phase_header "Step 4b SKIPPED: train suppressor"
else
    phase_header "Step 4b: train suppressor paired-DPO LoRA (v=${DEST_VERSION})"
    if ! train_one_direction suppressor; then
        echo "!!! Step 4b (sup train) FAILED."
        exit 1
    fi
fi

# Step 5b: validate suppressor
if [ "${SKIP_VAL_SUP:-0}" = "1" ]; then
    phase_header "Step 5b SKIPPED: validate suppressor"
else
    phase_header "Step 5b: validate suppressor persona LoRA (n=${N_PERSONAS})"
    if ! validate_one_direction suppressor "$CONST_STEM_SUP" sup; then
        echo "!!! Step 5b (sup validate) FAILED."
        VAL_FAILED+=("conviction_sup_v5")
    fi
fi

phase_header "All conviction v5 steps complete."
echo
if [ ${#VAL_FAILED[@]} -gt 0 ]; then
    echo "Validation failures: ${VAL_FAILED[*]}"
fi
echo "Summaries:"
echo "  v5 persona AMP:    scratch/factor_inspect/validate/${TRAIT}_amp_v5/${TRAIT}_amp_v5_summary.json"
echo "  v5 persona SUP:    scratch/factor_inspect/validate/${TRAIT}_sup_v5/${TRAIT}_sup_v5_summary.json"
echo
echo "Trained v5 adapters on monorepo:"
echo "  fine_tuning/llama-3.1-8b-it/unsupervised/${TRAIT}/{amplifier,suppressor}/v${DEST_VERSION}/lora/"
echo
echo "Overall log: ${OVERALL_LOG}"
