#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Overnight pipeline for the unsup_4fac F0 (Conviction) v2 retrain.
#
# Sequence (single GPU, sequential):
#
#   1. DPO-only validation of the existing v1 (paired_dpo) adapters
#      → answers "did SFT introduce the wrong-direction shift?" before we
#      commit to retraining.
#
#   2. Phase 1: teacher distillation with K=5 samples per prompt, using the
#      rewritten conviction constitution (FORMAT-clarified F0+/F0- prose).
#      Writes to monorepo at v<SOURCE_VERSION>.
#
#   3. Phase 2: seed paired-DPO data with --amp-pairing all (K^2 pairs/prompt
#      = max signal). Writes to monorepo at v<DEST_VERSION>.
#
#   4. Phase 3: full training (DPO + introspection + SFT + merge) for both
#      amplifier and suppressor.
#
#   5. Post-train validation of the new v2 persona LoRAs.
#
# Single GPU, run via tmux. See bottom of file for the tmux command.
#
# Usage:
#   bash scripts_dev/oct_pipeline/unsup_4fac/run_overnight_conviction_v2.sh <gpu_id>
#
# Skip phases via env var (set to 1 to skip):
#   SKIP_DPO_VALIDATE=1   skip step 1
#   SKIP_DISTILL=1        skip step 2
#   SKIP_SEED=1           skip step 3
#   SKIP_TRAIN=1          skip step 4
#   SKIP_VALIDATE=1       skip step 5
# ─────────────────────────────────────────────────────────────────────────────
set -o pipefail

GPU="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
export MASTER_PORT="$((29500 + GPU))"
export PYTHONUNBUFFERED=1

# ── Pipeline parameters ────────────────────────────────────────────────────
TRAIT="conviction"
SOURCE_VERSION="unsup_4fac_v2"            # Phase 1 distillation lands here
DEST_VERSION="unsup_4fac_paired_dpo_v2"   # Phase 2/3 paired-DPO + adapters land here
TEACHER_K=3                                # samples per prompt in distillation
AMP_PAIRING="all"                          # K^2 pairs per prompt for max signal
N_PERSONAS=200                             # validation subsample size

REPO_ROOT="/root/persona-shattering-lasr"
LOG_DIR="${REPO_ROOT}/scratch/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OVERALL_LOG="${LOG_DIR}/overnight_conviction_v2_${STAMP}.log"

phase_header() {
    echo
    echo "################################################################"
    echo "# $1"
    echo "# $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "################################################################"
}

# ── Step 1: DPO-only validation of the existing v1 adapters ────────────────
if [ "${SKIP_DPO_VALIDATE:-0}" = "1" ]; then
    phase_header "Step 1 SKIPPED: DPO-only validation of v1 adapters"
else
    phase_header "Step 1: DPO-only validation of v1 adapters"
    if ! bash "${REPO_ROOT}/scripts_dev/oct_pipeline/unsup_4fac/run_validate_conviction_dpo_only.sh" "$GPU"; then
        echo "!!! Step 1 FAILED — bailing before re-training."
        exit 1
    fi
fi

# ── Step 2: Phase 1 distillation with K=5, new constitution ────────────────
if [ "${SKIP_DISTILL:-0}" = "1" ]; then
    phase_header "Step 2 SKIPPED: distillation"
else
    phase_header "Step 2: Phase 1 distillation (K=${TEACHER_K}, v=${SOURCE_VERSION})"
    if ! VERSION="$SOURCE_VERSION" TEACHER_K="$TEACHER_K" \
            bash "${REPO_ROOT}/scripts_dev/oct_pipeline/unsup_4fac/prep_unsup_4fac_distillation.sh" \
            "$GPU" "$TRAIT"; then
        echo "!!! Step 2 FAILED — bailing."
        exit 1
    fi
fi

# ── Step 3: Phase 2 seed paired-DPO (amp_pairing=all) ──────────────────────
if [ "${SKIP_SEED:-0}" = "1" ]; then
    phase_header "Step 3 SKIPPED: seed"
else
    phase_header "Step 3: Phase 2 paired-DPO seed (amp_pairing=${AMP_PAIRING}, src=v${SOURCE_VERSION}, dest=v${DEST_VERSION})"
    if ! SOURCE_VERSION="$SOURCE_VERSION" DEST_VERSION="$DEST_VERSION" AMP_PAIRING="$AMP_PAIRING" \
            bash "${REPO_ROOT}/scripts_dev/oct_pipeline/unsup_4fac/seed_unsup_4fac_paired_dpo.sh" "$TRAIT"; then
        echo "!!! Step 3 FAILED — bailing."
        exit 1
    fi
fi

# ── Step 4: Phase 3 training (DPO + introspection + SFT + merge) ───────────
if [ "${SKIP_TRAIN:-0}" = "1" ]; then
    phase_header "Step 4 SKIPPED: training"
else
    phase_header "Step 4: Phase 3 training (v=${DEST_VERSION})"
    if ! VERSION="$DEST_VERSION" \
            bash "${REPO_ROOT}/scripts_dev/oct_pipeline/unsup_4fac/run_unsup_4fac_paired_dpo.sh" \
            "$GPU" "$TRAIT"; then
        echo "!!! Step 4 FAILED — bailing before validation."
        exit 1
    fi
fi

# ── Step 5: post-train validation of the v2 persona LoRAs ──────────────────
if [ "${SKIP_VALIDATE:-0}" = "1" ]; then
    phase_header "Step 5 SKIPPED: post-train validation"
else
    phase_header "Step 5: Post-train validation of v2 amp + sup persona LoRAs"

    AMP_ADAPTER="persona-shattering-lasr/monorepo::fine_tuning/llama-3.1-8b-it/unsupervised/${TRAIT}/amplifier/v${DEST_VERSION}/lora/${TRAIT}_amplifying_full_unsup_4fac-persona"
    SUP_ADAPTER="persona-shattering-lasr/monorepo::fine_tuning/llama-3.1-8b-it/unsupervised/${TRAIT}/suppressor/v${DEST_VERSION}/lora/${TRAIT}_suppressing_full_unsup_4fac-persona"

    AMP_LABEL="${TRAIT}_amp_v2"
    SUP_LABEL="${TRAIT}_sup_v2"

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

phase_header "All overnight steps complete."
echo
echo "Summaries:"
echo "  v1 DPO-only AMP:   scratch/factor_inspect/validate/conviction_amp_dpo_only/conviction_amp_dpo_only_summary.json"
echo "  v1 DPO-only SUP:   scratch/factor_inspect/validate/conviction_sup_dpo_only/conviction_sup_dpo_only_summary.json"
echo "  v2 persona AMP:    scratch/factor_inspect/validate/${TRAIT}_amp_v2/${TRAIT}_amp_v2_summary.json"
echo "  v2 persona SUP:    scratch/factor_inspect/validate/${TRAIT}_sup_v2/${TRAIT}_sup_v2_summary.json"
echo
echo "Trained v2 adapters on monorepo:"
echo "  fine_tuning/llama-3.1-8b-it/unsupervised/${TRAIT}/{amplifier,suppressor}/v${DEST_VERSION}/lora/"
echo
echo "Overall log: ${OVERALL_LOG}"
