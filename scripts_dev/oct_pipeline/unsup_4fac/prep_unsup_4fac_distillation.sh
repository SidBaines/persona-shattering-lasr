#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Run teacher distillation (only) for both amplifier and suppressor
# constitutions of a given unsup_4fac target trait, populating the monorepo
# with the JSONLs that prep_paired_dpo.py needs as inputs.
#
# Works for any unsup_4fac target with constitution files in this directory
# named "<trait>_amplifying_full_unsup_4fac.json" / "<trait>_suppressing_..."
# (currently warmth and conviction).
#
# After this script finishes, the monorepo will contain:
#   fine_tuning/llama-3.1-8b-it/unsupervised/<trait>/amplifier/vunsup_4fac/
#       data/distillation/<trait>_amplifying_full_unsup_4fac.jsonl
#   fine_tuning/llama-3.1-8b-it/unsupervised/<trait>/suppressor/vunsup_4fac/
#       data/distillation/<trait>_suppressing_full_unsup_4fac.jsonl
# (plus distillation_generation stage markers).
#
# Usage:
#   bash scripts_dev/oct_pipeline/unsup_4fac/prep_unsup_4fac_distillation.sh <gpu_id> <trait>
#
#   <trait> ∈ {warmth, conviction, exuberance, didacticism}
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

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
# Override with VERSION=... env var to write to a different monorepo subpath.
VERSION="${VERSION:-unsup_4fac}"   # MonorepoConfig prepends 'v' → fine_tuning/.../vunsup_4fac/
# Override with TEACHER_K=... env var to request K teacher samples per prompt
# (default empty = upstream default = 1). Larger K → more DPO signal downstream.
TEACHER_K="${TEACHER_K:-}"
# Override with CONST_STEM_AMP / CONST_STEM_SUP env vars to use non-default
# constitution filenames (e.g. v3 clement-style files). Defaults preserve
# the v1/v2 ``_full_unsup_4fac`` naming convention.
CONST_STEM_AMP="${CONST_STEM_AMP:-${TRAIT}_amplifying_full_unsup_4fac}"
CONST_STEM_SUP="${CONST_STEM_SUP:-${TRAIT}_suppressing_full_unsup_4fac}"
# Set CONCAT_ALL_TRAITS=1 to pass --concat-all-traits-system-prompt through
# to the OCT pipeline (clement-style constitutions need this).
CONCAT_ALL_TRAITS="${CONCAT_ALL_TRAITS:-0}"

LOG_DIR="scratch/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

run_distillation() {
    local DIRECTION="$1"      # amplifier | suppressor
    local CONST_NAME="$2"     # <trait>_amplifying_full_unsup_4fac (no .json)
    local OUT_DIR="scratch/oct_unsup_4fac_${TRAIT}_${DIRECTION}_${VERSION}_distill"
    local CONST_JSON="scripts_dev/oct_pipeline/unsup_4fac/${CONST_NAME}.json"
    local RUN_LOG="${LOG_DIR}/unsup_4fac_${TRAIT}_${DIRECTION}_${VERSION}_distill_${STAMP}.log"

    if [ ! -f "$CONST_JSON" ]; then
        echo "ERROR: constitution file not found: $CONST_JSON" >&2
        exit 1
    fi

    echo "================================================================"
    echo "  ${TRAIT} ${DIRECTION} — distillation only"
    echo "  GPU:           ${GPU}"
    echo "  constitution:  ${CONST_JSON}"
    echo "  out_dir:       ${OUT_DIR}"
    echo "  log:           ${RUN_LOG}"
    echo "================================================================"

    # `--stages distillation --skip-training --skip-student-distillation`:
    # Run only the teacher distillation generation (and upload to monorepo).
    # No student baseline pass (its output is unused for paired-DPO — the
    # seed step overwrites the student column with the rejected teacher's
    # response), no DPO-pair conversion, no DPO/SFT/merge training.
    local TEACHER_K_FLAG=()
    if [ -n "$TEACHER_K" ]; then
        TEACHER_K_FLAG=(--teacher-k "$TEACHER_K")
        echo "  teacher-k:     ${TEACHER_K}"
    fi
    local CONCAT_FLAG=()
    if [ "$CONCAT_ALL_TRAITS" = "1" ]; then
        CONCAT_FLAG=(--concat-all-traits-system-prompt)
        echo "  concat-all-traits-system-prompt: yes"
    fi
    {
      printf 'y\n' | uv run --with-requirements scripts_dev/oct_pipeline/uv-oct-requirements.txt \
        python scripts_dev/oct_pipeline/run_oct_pipeline.py \
          --model "$MODEL" \
          --teacher-model "$TEACHER" \
          --custom-constitution "$CONST_JSON" \
          --out-dir "$OUT_DIR" \
          --monorepo-category unsupervised \
          --monorepo-trait "$TRAIT" \
          --monorepo-direction "$DIRECTION" \
          --monorepo-version "$VERSION" \
          "${TEACHER_K_FLAG[@]}" \
          "${CONCAT_FLAG[@]}" \
          --stages distillation \
          --skip-training \
          --skip-student-distillation
    } 2>&1 | tee "$RUN_LOG"

    echo "  ✓ ${DIRECTION} distillation complete"
}

run_distillation amplifier  "${CONST_STEM_AMP}"
run_distillation suppressor "${CONST_STEM_SUP}"

echo
echo "================================================================"
echo "  Phase 1 done. Next:"
echo "    bash scripts_dev/oct_pipeline/unsup_4fac/seed_unsup_4fac_paired_dpo.sh ${TRAIT}"
echo "================================================================"
