#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Overnight pipeline: v6 conviction — SINGLE-TEACHER (chosen=teacher,
# rejected=Llama-baseline) DPO. Comparison point for the paired-teacher v6
# run (run_overnight_conviction_v6.sh, where chosen and rejected are both
# teacher samples under opposite-pole constitutions).
#
# This script reuses the teacher + student distillation that the paired-DPO
# Phase 1 already generated and uploaded to monorepo at
#   fine_tuning/llama-3.1-8b-it/unsupervised/conviction/{amp,sup}/
#       vunsup_4fac_v6/data/distillation/<const_stem>.jsonl
# Each of those JSONLs has columns: prompt, response (teacher under v6
# amp/sup constitution), llama-3.1-8b-it (Llama 3.1-8B baseline). The
# OCT pipeline's load_dpo_pairs reads chosen=response, rejected=llama-…,
# which is the standard single-teacher DPO setup.
#
# So this script just calls the existing run_unsup_4fac_paired_dpo.sh with
# VERSION=unsup_4fac_v6 (i.e., write LoRA adapters back to the same monorepo
# subtree that holds the source distillation). Despite that script's name,
# it is just a wrapper around run_oct_pipeline.py — when no paired-DPO seed
# has been done at the target VERSION, the pipeline does standard
# single-teacher DPO using the cached teacher + student distillation.
#
# Sequence, single GPU — interleaved per pole so we can read amp results
# while sup is still training:
#   1. Generate/check v6 conviction constitutions (idempotent).
#   2a. Train amplifier LoRA (DPO only by default; full pipeline if STAGES=all).
#   3a. Validate amplifier {-dpo by default | -persona if STAGES=all} LoRA.
#   2b. Train suppressor LoRA.
#   3b. Validate suppressor LoRA.
#
# No Phase 2 distillation (already cached). No Phase 3 paired-DPO seed.
#
# Usage:
#   bash scripts_dev/oct_pipeline/unsup_4fac/run_overnight_conviction_v6_singleteacher.sh <gpu_id>
#
# To run the full pipeline (DPO + introspection + SFT + merge, ~3-5x slower):
#   STAGES=all bash .../run_overnight_conviction_v6_singleteacher.sh <gpu_id>
#
# Skip flags (set to 1 to skip):
#   SKIP_GENERATE=1   skip constitution generation/check
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
# Single-teacher DPO writes LoRA adapters back to the same monorepo path
# that holds the source distillation (vunsup_4fac_v6). Override only if
# you've staged the same teacher+student distillation under a different
# version path.
VERSION="${VERSION:-unsup_4fac_v6}"
TEACHER_K="${TEACHER_K:-}"
N_PERSONAS="${N_PERSONAS:-200}"

# Default: DPO-only training for fast turnaround. STAGES=all does
# distillation (cached, will skip) + DPO + introspection + SFT + merge.
STAGES="${STAGES:-distillation}"

case "$STAGES" in
    distillation) DEFAULT_ADAPTER_KIND=dpo ;;
    *)            DEFAULT_ADAPTER_KIND=persona ;;
esac
ADAPTER_KIND="${ADAPTER_KIND:-$DEFAULT_ADAPTER_KIND}"

CONST_STEM_AMP="conviction_amplifying_v6_unsup_4fac"
CONST_STEM_SUP="conviction_suppressing_v6_unsup_4fac"

REPO_ROOT="/root/persona-shattering-lasr"
LOG_DIR="${REPO_ROOT}/scratch/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OVERALL_LOG="${LOG_DIR}/overnight_conviction_v6_singleteacher_${STAMP}.log"

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
        VERSION="$VERSION" \
        CONST_STEM_AMP="$CONST_STEM_AMP" CONST_STEM_SUP="$CONST_STEM_SUP" \
        CONCAT_ALL_TRAITS=0 \
        STAGES="$STAGES" \
        bash "${REPO_ROOT}/scripts_dev/oct_pipeline/unsup_4fac/run_unsup_4fac_paired_dpo.sh" \
        "$GPU" "$TRAIT"
}

validate_one_direction() {
    # $1: "amplifier" or "suppressor"; $2: const stem; $3: short label suffix
    local DIR="$1"
    local STEM="$2"
    local SUFFIX="$3"
    local ADAPTER="persona-shattering-lasr/monorepo::fine_tuning/llama-3.1-8b-it/unsupervised/${TRAIT}/${DIR}/v${VERSION}/lora/${STEM}-${ADAPTER_KIND}"
    # Label includes "_singleteacher" to keep validation outputs separate
    # from the paired-DPO ones (and includes the adapter kind for distinct
    # on-disk dirs).
    local LABEL
    if [ "$ADAPTER_KIND" = "persona" ]; then
        LABEL="${TRAIT}_${SUFFIX}_v6_singleteacher"
    else
        LABEL="${TRAIT}_${SUFFIX}_${ADAPTER_KIND}_v6_singleteacher"
    fi
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

# Step 1: constitutions (idempotent — re-emits identical JSONs)
if [ "${SKIP_GENERATE:-0}" = "1" ]; then
    phase_header "Step 1 SKIPPED: generate/check v6 constitutions"
else
    phase_header "Step 1: generate/check v6 constitutions"
    if ! uv run python "${REPO_ROOT}/scripts_dev/oct_pipeline/unsup_4fac/generate_v6_conviction_constitutions.py"; then
        echo "!!! Step 1 FAILED."
        exit 1
    fi
fi

VAL_FAILED=()

# Pre-compute label names so failure tracking + summary block can use them
if [ "$ADAPTER_KIND" = "persona" ]; then
    AMP_LABEL="${TRAIT}_amp_v6_singleteacher"
    SUP_LABEL="${TRAIT}_sup_v6_singleteacher"
else
    AMP_LABEL="${TRAIT}_amp_${ADAPTER_KIND}_v6_singleteacher"
    SUP_LABEL="${TRAIT}_sup_${ADAPTER_KIND}_v6_singleteacher"
fi

# Step 2a: train amplifier (single-teacher DPO)
if [ "${SKIP_TRAIN_AMP:-0}" = "1" ]; then
    phase_header "Step 2a SKIPPED: train amplifier"
else
    phase_header "Step 2a: train amplifier single-teacher DPO LoRA (v=${VERSION})"
    if ! train_one_direction amplifier; then
        echo "!!! Step 2a (amp train) FAILED."
        exit 1
    fi
fi

# Step 3a: validate amplifier
if [ "${SKIP_VAL_AMP:-0}" = "1" ]; then
    phase_header "Step 3a SKIPPED: validate amplifier"
else
    phase_header "Step 3a: validate amplifier ${ADAPTER_KIND} LoRA (n=${N_PERSONAS})"
    if ! validate_one_direction amplifier "$CONST_STEM_AMP" amp; then
        echo "!!! Step 3a (amp validate) FAILED — continuing to sup."
        VAL_FAILED+=("$AMP_LABEL")
    fi
fi

# Step 2b: train suppressor
if [ "${SKIP_TRAIN_SUP:-0}" = "1" ]; then
    phase_header "Step 2b SKIPPED: train suppressor"
else
    phase_header "Step 2b: train suppressor single-teacher DPO LoRA (v=${VERSION})"
    if ! train_one_direction suppressor; then
        echo "!!! Step 2b (sup train) FAILED."
        exit 1
    fi
fi

# Step 3b: validate suppressor
if [ "${SKIP_VAL_SUP:-0}" = "1" ]; then
    phase_header "Step 3b SKIPPED: validate suppressor"
else
    phase_header "Step 3b: validate suppressor ${ADAPTER_KIND} LoRA (n=${N_PERSONAS})"
    if ! validate_one_direction suppressor "$CONST_STEM_SUP" sup; then
        echo "!!! Step 3b (sup validate) FAILED."
        VAL_FAILED+=("$SUP_LABEL")
    fi
fi

phase_header "All conviction v6 single-teacher steps complete."
echo
echo "Mode: STAGES=${STAGES}  ADAPTER_KIND=${ADAPTER_KIND}  (single-teacher DPO)"
if [ ${#VAL_FAILED[@]} -gt 0 ]; then
    echo "Validation failures: ${VAL_FAILED[*]}"
fi
echo "Summaries:"
echo "  v6 single-teacher AMP:    scratch/factor_inspect/validate/${AMP_LABEL}/${AMP_LABEL}_summary.json"
echo "  v6 single-teacher SUP:    scratch/factor_inspect/validate/${SUP_LABEL}/${SUP_LABEL}_summary.json"
echo
echo "Trained v6 single-teacher adapters on monorepo:"
echo "  fine_tuning/llama-3.1-8b-it/unsupervised/${TRAIT}/{amplifier,suppressor}/v${VERSION}/lora/{${CONST_STEM_AMP},${CONST_STEM_SUP}}-${ADAPTER_KIND}"
echo
echo "Overall log: ${OVERALL_LOG}"
