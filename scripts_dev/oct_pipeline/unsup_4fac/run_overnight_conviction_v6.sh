#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Overnight pipeline for the unsup_4fac F0 (Conviction) v6 retrain.
#
# v6 = constitution rewrite around the **engaged-agency stance** framing:
#   * 4 facets (Belief Verification, Stable Position, Charitable Pushback,
#     Engaged-with-Stakes) explicitly targeting all three behaviorally-
#     trainable sub-axes of F0 — including the conscientious-action-
#     recommendation MCQ pattern (Sub-axis B) which v1/v3/v5 never
#     touched.
#   * Vanton4-style structural layout: single full-constitution file per
#     pole + slim variant for introspection. 8 entries per full file
#     (4 facets × 2 framings), 50 questions per entry, 200 unique prompts
#     shared between amp and sup.
#   * Drops F1/F3-bait language vs v1: no FORMAT prescription, no
#     "anticipate followups + volunteer related topics" framing, no
#     Anticipatory-Context facet.
#   * Resolves the Sub-axis A vs Sub-axis C tension (acquiescence-
#     contaminated yielding Likerts) explicitly in the Stable Position
#     facet's prose: verify the challenge first, then hold position if it
#     doesn't pan out.
#
# Pipeline matches v5's overnight runner (and the OCEAN vanton4 paired-DPO
# recipe): teacher_k=1, amp_pairing=first, no concat-all-traits flag.
#
# DEFAULT MODE: DPO-only training for fast turnaround. Skips introspection,
# SFT, and merge stages — produces only the {STEM}-dpo adapter, validated
# directly. Override with STAGES=all for the full pipeline (validates the
# {STEM}-persona SFT-merged adapter instead). DPO-only is ~3-5x faster
# than the full pipeline.
#
# Sequence, single GPU — interleaved per pole so we can read amp results
# while sup is still training:
#   1. Generate/check v6 conviction constitutions.
#   2. Teacher distillation for amplifier and suppressor (K=1, no concat).
#   3. Seed paired-DPO data (amp_pairing=first).
#   4a. Train amplifier LoRA (DPO only by default; full pipeline if STAGES=all).
#   5a. Validate amplifier {-dpo by default | -persona if STAGES=all} LoRA.
#   4b. Train suppressor LoRA (DPO only by default; full pipeline if STAGES=all).
#   5b. Validate suppressor {-dpo by default | -persona if STAGES=all} LoRA.
#
# Usage:
#   bash scripts_dev/oct_pipeline/unsup_4fac/run_overnight_conviction_v6.sh <gpu_id>
#
# To run the full pipeline (DPO + SFT + merge, ~3-5x slower):
#   STAGES=all bash scripts_dev/oct_pipeline/unsup_4fac/run_overnight_conviction_v6.sh <gpu_id>
#
# Suggested tmux launch:
#   mkdir -p scratch/logs
#   LOG=scratch/logs/overnight_conviction_v6_$(date -u +%Y%m%dT%H%M%SZ).log
#   tmux new -d -s conviction_v6 "cd /root/persona-shattering-lasr && bash scripts_dev/oct_pipeline/unsup_4fac/run_overnight_conviction_v6.sh 0 2>&1 | tee $LOG"
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
SOURCE_VERSION="${SOURCE_VERSION:-unsup_4fac_v6}"
DEST_VERSION="${DEST_VERSION:-unsup_4fac_paired_dpo_v6}"
TEACHER_K="${TEACHER_K:-}"          # empty ⇒ k=1 (omits --teacher-k flag)
AMP_PAIRING="${AMP_PAIRING:-first}" # 1 pair per prompt, like vanton4 + v5
N_PERSONAS="${N_PERSONAS:-200}"

# Default: DPO-only training for fast turnaround. Skips introspection +
# SFT + merge; produces only the {STEM}-dpo adapter, which we validate
# directly. Set STAGES=all to do the full distillation→DPO→introspection
# →SFT→merge pipeline (~3-5x slower; produces -persona adapter for
# end-to-end validation against the SFT-merged model).
STAGES="${STAGES:-distillation}"

# Adapter checkpoint to validate. Defaults follow STAGES:
#   STAGES=distillation -> ADAPTER_KIND=dpo     (validate the DPO LoRA)
#   STAGES=all          -> ADAPTER_KIND=persona (validate the SFT-merged LoRA)
case "$STAGES" in
    distillation) DEFAULT_ADAPTER_KIND=dpo ;;
    *)            DEFAULT_ADAPTER_KIND=persona ;;
esac
ADAPTER_KIND="${ADAPTER_KIND:-$DEFAULT_ADAPTER_KIND}"

# v6 constitutions: 4 facets, vanton4-style trait bodies, F3-bait dropped,
# A/C tension resolved in Stable Position prose.
CONST_STEM_AMP="conviction_amplifying_v6_unsup_4fac"
CONST_STEM_SUP="conviction_suppressing_v6_unsup_4fac"

# Slim variants exist for introspection (defaults of run_unsup_4fac_paired_dpo.sh
# expect ${STEM}_slim, which matches the files emitted by the v6 generator).

REPO_ROOT="/root/persona-shattering-lasr"
LOG_DIR="${REPO_ROOT}/scratch/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OVERALL_LOG="${LOG_DIR}/overnight_conviction_v6_${STAMP}.log"

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
    # Adapter kind suffix follows ADAPTER_KIND (dpo when STAGES=distillation,
    # persona when STAGES=all).
    local ADAPTER="persona-shattering-lasr/monorepo::fine_tuning/llama-3.1-8b-it/unsupervised/${TRAIT}/${DIR}/v${DEST_VERSION}/lora/${STEM}-${ADAPTER_KIND}"
    # Label includes the adapter kind for distinct on-disk output dirs (e.g.
    # conviction_amp_dpo_v6 vs conviction_amp_v6 if both modes are run).
    local LABEL
    if [ "$ADAPTER_KIND" = "persona" ]; then
        LABEL="${TRAIT}_${SUFFIX}_v6"
    else
        LABEL="${TRAIT}_${SUFFIX}_${ADAPTER_KIND}_v6"
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

# Step 1: constitutions
if [ "${SKIP_GENERATE:-0}" = "1" ]; then
    phase_header "Step 1 SKIPPED: generate/check v6 constitutions"
else
    phase_header "Step 1: generate/check v6 constitutions"
    if ! uv run python "${REPO_ROOT}/scripts_dev/oct_pipeline/unsup_4fac/generate_v6_conviction_constitutions.py"; then
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
if [ "$ADAPTER_KIND" = "persona" ]; then
    AMP_LABEL="${TRAIT}_amp_v6"
    SUP_LABEL="${TRAIT}_sup_v6"
else
    AMP_LABEL="${TRAIT}_amp_${ADAPTER_KIND}_v6"
    SUP_LABEL="${TRAIT}_sup_${ADAPTER_KIND}_v6"
fi

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
    phase_header "Step 5b: validate suppressor ${ADAPTER_KIND} LoRA (n=${N_PERSONAS})"
    if ! validate_one_direction suppressor "$CONST_STEM_SUP" sup; then
        echo "!!! Step 5b (sup validate) FAILED."
        VAL_FAILED+=("$SUP_LABEL")
    fi
fi

phase_header "All conviction v6 steps complete."
echo
echo "Mode: STAGES=${STAGES}  ADAPTER_KIND=${ADAPTER_KIND}"
if [ ${#VAL_FAILED[@]} -gt 0 ]; then
    echo "Validation failures: ${VAL_FAILED[*]}"
fi
echo "Summaries:"
echo "  v6 ${ADAPTER_KIND} AMP:    scratch/factor_inspect/validate/${AMP_LABEL}/${AMP_LABEL}_summary.json"
echo "  v6 ${ADAPTER_KIND} SUP:    scratch/factor_inspect/validate/${SUP_LABEL}/${SUP_LABEL}_summary.json"
echo
echo "Trained v6 adapters on monorepo:"
echo "  fine_tuning/llama-3.1-8b-it/unsupervised/${TRAIT}/{amplifier,suppressor}/v${DEST_VERSION}/lora/"
echo
echo "Overall log: ${OVERALL_LOG}"
