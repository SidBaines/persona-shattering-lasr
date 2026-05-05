#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Overnight pipeline for the unsup_k4_v7_pf3 F1 (Pedagogy) paired-DPO LoRA.
#
# Same shape as run_overnight_initiative.sh — only TRAIT and labels differ.
# Sequence (single GPU, interleaved per pole):
#   1. Generate / check Pedagogy constitutions.
#   2. Teacher distillation for amplifier and suppressor (K=1).
#   3. Seed paired-DPO data (amp_pairing=first).
#   4a. Train amplifier LoRA (DPO + introspection + SFT + merge).
#   5a. Validate amplifier on 200 personas (re-admin v7 fc_pair).
#   4b. Train suppressor LoRA (DPO + introspection + SFT + merge).
#   5b. Validate suppressor on 200 personas (re-admin v7 fc_pair).
#
# Usage:
#   bash scripts_dev/oct_pipeline/unsup_k4_v7_pf3/run_overnight_pedagogy.sh <gpu_id>
#
# Suggested tmux launch:
#   mkdir -p scratch/logs
#   LOG=scratch/logs/overnight_pedagogy_$(date -u +%Y%m%dT%H%M%SZ).log
#   tmux new -d -s pedagogy_overnight \
#     "cd /root/persona-shattering && \
#      bash scripts_dev/oct_pipeline/unsup_k4_v7_pf3/run_overnight_pedagogy.sh 0 2>&1 | tee $LOG"
#
# Skip phases via env var (set to 1 to skip):
#   SKIP_GENERATE / SKIP_DISTILL / SKIP_SEED /
#   SKIP_TRAIN_AMP / SKIP_VAL_AMP / SKIP_TRAIN_SUP / SKIP_VAL_SUP
# ─────────────────────────────────────────────────────────────────────────────
set -o pipefail

GPU="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
export MASTER_PORT="$((29500 + GPU))"
export PYTHONUNBUFFERED=1

TRAIT="pedagogy"
SOURCE_VERSION="${SOURCE_VERSION:-unsup_k4_v7_pf3}"
DEST_VERSION="${DEST_VERSION:-unsup_k4_v7_pf3_paired_dpo}"
TEACHER_K="${TEACHER_K:-}"
AMP_PAIRING="${AMP_PAIRING:-first}"
N_PERSONAS="${N_PERSONAS:-200}"

# Default: full pipeline (DPO + introspection + SFT + merge).
STAGES="${STAGES:-all}"

case "$STAGES" in
    distillation) DEFAULT_ADAPTER_KIND=dpo ;;
    *)            DEFAULT_ADAPTER_KIND=persona ;;
esac
ADAPTER_KIND="${ADAPTER_KIND:-$DEFAULT_ADAPTER_KIND}"

CONST_STEM_AMP="${TRAIT}_amplifier"
CONST_STEM_SUP="${TRAIT}_suppressor"

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
LOG_DIR="${REPO_ROOT}/scratch/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OVERALL_LOG="${LOG_DIR}/overnight_${TRAIT}_${STAMP}.log"

phase_header() {
    echo
    echo "################################################################"
    echo "# $1"
    echo "# $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "################################################################"
}

train_one_direction() {
    local DIR="$1"
    DIRECTIONS_TO_RUN="$DIR" \
        VERSION="$DEST_VERSION" \
        CONST_STEM_AMP="$CONST_STEM_AMP" CONST_STEM_SUP="$CONST_STEM_SUP" \
        STAGES="$STAGES" \
        bash "${REPO_ROOT}/scripts_dev/oct_pipeline/unsup_k4_v7_pf3/run_unsup_k4_v7_pf3_paired_dpo.sh" \
        "$GPU" "$TRAIT"
}

validate_one_direction() {
    local DIR="$1"
    local STEM="$2"
    local SUFFIX="$3"
    local ADAPTER="persona-shattering-lasr/monorepo::fine_tuning/llama-3.1-8b-it/unsupervised/${TRAIT}/${DIR}/v${DEST_VERSION}/lora/${STEM}-${ADAPTER_KIND}"
    local LABEL
    if [ "$ADAPTER_KIND" = "persona" ]; then
        LABEL="${TRAIT}_${SUFFIX}"
    else
        LABEL="${TRAIT}_${SUFFIX}_${ADAPTER_KIND}"
    fi
    local VAL_LOG="${LOG_DIR}/${LABEL}_validate_${STAMP}.log"

    echo
    echo "  validating ${LABEL}"
    echo "    adapter: ${ADAPTER}"
    echo "    log:     ${VAL_LOG}"
    stdbuf -oL -eL uv run python \
        "${REPO_ROOT}/scripts_dev/oct_pipeline/unsup_k4_v7_pf3/validate_lora.py" \
        --target "$TRAIT" \
        --adapter "$ADAPTER" \
        --n-personas "$N_PERSONAS" \
        --label "$LABEL" \
        --direction "$DIR" \
        --monorepo-version "$DEST_VERSION" \
        --upload-monorepo \
        2>&1 | stdbuf -oL -eL tee "$VAL_LOG"
}

# Step 1: constitutions
if [ "${SKIP_GENERATE:-0}" = "1" ]; then
    phase_header "Step 1 SKIPPED: generate/check Pedagogy constitutions"
else
    phase_header "Step 1: generate/check Pedagogy constitutions"
    if ! uv run python "${REPO_ROOT}/scripts_dev/oct_pipeline/unsup_k4_v7_pf3/generate_pedagogy_constitutions.py"; then
        echo "!!! Step 1 FAILED."
        exit 1
    fi
fi

# Step 2: distillation
if [ "${SKIP_DISTILL:-0}" = "1" ]; then
    phase_header "Step 2 SKIPPED: distillation"
else
    phase_header "Step 2: distillation (K=${TEACHER_K:-1}, v=${SOURCE_VERSION})"
    if ! VERSION="$SOURCE_VERSION" TEACHER_K="$TEACHER_K" \
            CONST_STEM_AMP="$CONST_STEM_AMP" CONST_STEM_SUP="$CONST_STEM_SUP" \
            bash "${REPO_ROOT}/scripts_dev/oct_pipeline/unsup_k4_v7_pf3/prep_unsup_k4_v7_pf3_distillation.sh" \
            "$GPU" "$TRAIT"; then
        echo "!!! Step 2 FAILED."
        exit 1
    fi
fi

# Step 3: paired-DPO seed
if [ "${SKIP_SEED:-0}" = "1" ]; then
    phase_header "Step 3 SKIPPED: seed paired-DPO"
else
    phase_header "Step 3: seed paired-DPO (amp_pairing=${AMP_PAIRING}, src=v${SOURCE_VERSION}, dest=v${DEST_VERSION})"
    if ! SOURCE_VERSION="$SOURCE_VERSION" DEST_VERSION="$DEST_VERSION" AMP_PAIRING="$AMP_PAIRING" \
            CONST_STEM_AMP="$CONST_STEM_AMP" CONST_STEM_SUP="$CONST_STEM_SUP" \
            bash "${REPO_ROOT}/scripts_dev/oct_pipeline/unsup_k4_v7_pf3/seed_unsup_k4_v7_pf3_paired_dpo.sh" "$TRAIT"; then
        echo "!!! Step 3 FAILED."
        exit 1
    fi
fi

VAL_FAILED=()

if [ "$ADAPTER_KIND" = "persona" ]; then
    AMP_LABEL="${TRAIT}_amp"
    SUP_LABEL="${TRAIT}_sup"
else
    AMP_LABEL="${TRAIT}_amp_${ADAPTER_KIND}"
    SUP_LABEL="${TRAIT}_sup_${ADAPTER_KIND}"
fi

# Step 4a: train amplifier
if [ "${SKIP_TRAIN_AMP:-0}" = "1" ]; then
    phase_header "Step 4a SKIPPED: train amplifier"
else
    phase_header "Step 4a: train amplifier paired-DPO LoRA (v=${DEST_VERSION}, stages=${STAGES})"
    if ! train_one_direction amplifier; then
        echo "!!! Step 4a (amp train) FAILED."
        exit 1
    fi
fi

# Step 5a: validate amplifier
if [ "${SKIP_VAL_AMP:-0}" = "1" ]; then
    phase_header "Step 5a SKIPPED: validate amplifier"
else
    phase_header "Step 5a: validate amplifier ${ADAPTER_KIND} LoRA (n=${N_PERSONAS})"
    if ! validate_one_direction amplifier "$CONST_STEM_AMP" amp; then
        echo "!!! Step 5a (amp validate) FAILED — continuing to sup."
        VAL_FAILED+=("$AMP_LABEL")
    fi
fi

# Step 4b: train suppressor
if [ "${SKIP_TRAIN_SUP:-0}" = "1" ]; then
    phase_header "Step 4b SKIPPED: train suppressor"
else
    phase_header "Step 4b: train suppressor paired-DPO LoRA (v=${DEST_VERSION}, stages=${STAGES})"
    if ! train_one_direction suppressor; then
        echo "!!! Step 4b (sup train) FAILED."
        exit 1
    fi
fi

# Step 5b: validate suppressor
if [ "${SKIP_VAL_SUP:-0}" = "1" ]; then
    phase_header "Step 5b SKIPPED: validate suppressor"
else
    phase_header "Step 5b: validate suppressor ${ADAPTER_KIND} LoRA (n=${N_PERSONAS})"
    if ! validate_one_direction suppressor "$CONST_STEM_SUP" sup; then
        echo "!!! Step 5b (sup validate) FAILED."
        VAL_FAILED+=("$SUP_LABEL")
    fi
fi

phase_header "Overnight ${TRAIT} pipeline complete."
echo
echo "Mode: STAGES=${STAGES}  ADAPTER_KIND=${ADAPTER_KIND}"
if [ ${#VAL_FAILED[@]} -gt 0 ]; then
    echo "Validation failures: ${VAL_FAILED[*]}"
fi
echo "Summaries:"
echo "  ${ADAPTER_KIND} AMP:    scratch/factor_inspect_v7_pf3/validate/${AMP_LABEL}/${AMP_LABEL}_summary.json"
echo "  ${ADAPTER_KIND} SUP:    scratch/factor_inspect_v7_pf3/validate/${SUP_LABEL}/${SUP_LABEL}_summary.json"
echo
echo "Trained adapters on monorepo:"
echo "  fine_tuning/llama-3.1-8b-it/unsupervised/${TRAIT}/{amplifier,suppressor}/v${DEST_VERSION}/lora/"
echo
echo "Overall log: ${OVERALL_LOG}"
