#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Train unsup_4fac LoRAs (amplifier + suppressor) for a given target
# trait using paired-teacher DPO. Mirrors run_agreeableness_vanton4_paired_dpo.sh
# but for our unsup_4fac targets.
#
# Prereq: Phases 1 and 2 must have completed; the monorepo must contain
# paired-DPO distillation JSONLs at
#   fine_tuning/llama-3.1-8b-it/unsupervised/<trait>/{amplifier,suppressor}/
#       vunsup_4fac_paired_dpo/data/distillation/<const>.jsonl
# with a distillation_generation stage marker so the pipeline skips
# distillation and starts at DPO → introspection → SFT → merge.
#
# Trains both directions sequentially on a single GPU. Set DIRECTIONS_TO_RUN
# below to run only one if needed.
#
# Usage:
#   bash scripts_dev/oct_pipeline/unsup_4fac/run_unsup_4fac_paired_dpo.sh <gpu_id> <trait>
#
#   <trait> ∈ {warmth, conviction, exuberance, didacticism}
# ─────────────────────────────────────────────────────────────────────────────
set -o pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <gpu_id> <trait>" >&2
    echo "  <trait> ∈ {warmth, conviction, exuberance, didacticism}" >&2
    exit 2
fi

GPU="$1"
TRAIT="$2"

case "$TRAIT" in
    warmth|conviction|exuberance|didacticism) ;;
    *) echo "ERROR: unknown <trait> '$TRAIT' (expected one of warmth/conviction/exuberance/didacticism)" >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES="$GPU"
export MASTER_PORT="$((29500 + GPU))"

MODEL="llama-3.1-8b-it"
TEACHER="z-ai/glm-4.5-air"
# MonorepoConfig.path_prefix builds f"v{version}", so pass WITHOUT the leading
# "v" — uploads/reads land at .../v<VERSION>/. Override with VERSION=... env
# var to write to a different monorepo subpath (e.g. unsup_4fac_paired_dpo_v2).
VERSION="${VERSION:-unsup_4fac_paired_dpo}"

# H100 SXM (80 GB) throughput overrides — same as the OCEAN paired_dpo runs.
DPO_MICRO_BATCH=8
SFT_MICRO_BATCH=16
INTROSPECTION_MAX_NUM_SEQS=2048
INTROSPECTION_MAX_NUM_BATCHED_TOKENS=65536

LOG_DIR="scratch/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

FAILED=()

# Set to "amplifier suppressor" to train both, "amplifier" or "suppressor" for
# just one. The amplifier is the primary training target; the suppressor is
# trained as the validation companion (push the factor down on the same student).
DIRECTIONS_TO_RUN="amplifier suppressor"

for DIRECTION in $DIRECTIONS_TO_RUN; do
    if [ "$DIRECTION" = "amplifier" ]; then
        STEM="${TRAIT}_amplifying_full_unsup_4fac"
    else
        STEM="${TRAIT}_suppressing_full_unsup_4fac"
    fi
    CONST_JSON="scripts_dev/oct_pipeline/unsup_4fac/${STEM}.json"
    SLIM_JSON="scripts_dev/oct_pipeline/unsup_4fac/${STEM}_slim.json"
    OUT_DIR="scratch/oct_unsup_4fac_${TRAIT}_${DIRECTION}_${VERSION}"
    RUN_LOG="${LOG_DIR}/unsup_4fac_${TRAIT}_${DIRECTION}_${VERSION}_${STAMP}.log"

    if [ ! -f "$CONST_JSON" ]; then
        echo "ERROR: constitution file not found: $CONST_JSON" >&2
        FAILED+=("$DIRECTION (missing constitution)")
        continue
    fi
    if [ ! -f "$SLIM_JSON" ]; then
        echo "ERROR: slim constitution file not found: $SLIM_JSON" >&2
        FAILED+=("$DIRECTION (missing slim constitution)")
        continue
    fi

    echo
    echo "================================================================"
    echo "  ${TRAIT} ${DIRECTION} — paired-teacher DPO training"
    echo "  GPU:                     ${GPU}"
    echo "  constitution:            ${CONST_JSON}"
    echo "  introspection (slim):    ${SLIM_JSON}"
    echo "  out_dir:                 ${OUT_DIR}"
    echo "  log:                     ${RUN_LOG}"
    echo "  monorepo version:        ${VERSION}"
    echo "================================================================"

    if ! {
      printf 'y\n' | uv run --with-requirements scripts_dev/oct_pipeline/uv-oct-requirements.txt \
        python scripts_dev/oct_pipeline/run_oct_pipeline.py \
          --model "$MODEL" \
          --teacher-model "$TEACHER" \
          --custom-constitution "$CONST_JSON" \
          --introspection-constitution "$SLIM_JSON" \
          --out-dir "$OUT_DIR" \
          --monorepo-category unsupervised \
          --monorepo-trait "$TRAIT" \
          --monorepo-direction "$DIRECTION" \
          --monorepo-version "$VERSION" \
          --oct-dpo-micro-batch-size "$DPO_MICRO_BATCH" \
          --oct-sft-micro-batch-size "$SFT_MICRO_BATCH" \
          --introspection-max-num-seqs "$INTROSPECTION_MAX_NUM_SEQS" \
          --introspection-max-num-batched-tokens "$INTROSPECTION_MAX_NUM_BATCHED_TOKENS"
    } 2>&1 | tee "$RUN_LOG"; then
        echo "!!! FAILED: ${DIRECTION}"
        FAILED+=("$DIRECTION")
    else
        rm -rf "${OUT_DIR}/models/distilled/"
        echo "  ✓ ${DIRECTION} paired_dpo training complete"
    fi
done

echo
echo "================================================================"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "  Phase 3 done."
    echo "  Trained adapters live on monorepo at:"
    echo "    fine_tuning/llama-3.1-8b-it/unsupervised/${TRAIT}/{amp,sup}/v${VERSION}/lora/"
    echo "  Validate with:"
    echo "    uv run python scripts_dev/oct_pipeline/unsup_4fac/validate_lora.py --target ${TRAIT} ..."
else
    echo "  Phase 3 had failures:"
    for f in "${FAILED[@]}"; do echo "    - $f"; done
fi
echo "================================================================"
