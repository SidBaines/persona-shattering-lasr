#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Run teacher distillation (only) for both amplifier and suppressor
# constitutions of a given unsup_k4_v7_pf3 target trait, populating the
# monorepo with the JSONLs that prep_paired_dpo.py needs as inputs.
#
# Mirrors scripts_dev/oct_pipeline/unsup_4fac/prep_unsup_4fac_distillation.sh
# but for the k=4 v7_pf3 oblimin solution: distinct monorepo version
# (vunsup_k4_v7_pf3) so the artefacts don't collide with the v5-FA-based
# unsup_4fac runs that share trait names like "warmth".
#
# Works for any unsup_k4_v7_pf3 target with constitution files in this
# directory named "<trait>_amplifier.json" / "<trait>_suppressor.json".
# Currently only "initiative" is wired up; others (pedagogy, warmth,
# hedging) are added as their constitution JSONs land in this dir.
#
# After this script finishes, the monorepo will contain:
#   fine_tuning/llama-3.1-8b-it/unsupervised/<trait>/amplifier/vunsup_k4_v7_pf3/
#       data/distillation/<trait>_amplifier.jsonl
#   fine_tuning/llama-3.1-8b-it/unsupervised/<trait>/suppressor/vunsup_k4_v7_pf3/
#       data/distillation/<trait>_suppressor.jsonl
# (plus distillation_generation stage markers).
#
# Usage:
#   bash scripts_dev/oct_pipeline/unsup_k4_v7_pf3/prep_unsup_k4_v7_pf3_distillation.sh <gpu_id> <trait>
#
#   <trait> ∈ {initiative}    (extend when other constitutions land)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <gpu_id> <trait>" >&2
    echo "  <trait> ∈ {initiative}" >&2
    exit 2
fi

GPU="$1"
TRAIT="$2"

case "$TRAIT" in
    initiative|pedagogy|warmth|hedging) ;;
    *) echo "ERROR: unknown <trait> '$TRAIT' (expected one of initiative/pedagogy/warmth/hedging)" >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES="$GPU"
export MASTER_PORT="$((29500 + GPU))"

MODEL="llama-3.1-8b-it"
TEACHER="z-ai/glm-4.5-air"
# Monorepo version (without leading 'v'; MonorepoConfig prepends it). Override
# via VERSION=... env var to write to a different subpath.
VERSION="${VERSION:-unsup_k4_v7_pf3}"
# K teacher samples per prompt (default empty = upstream default = 1).
TEACHER_K="${TEACHER_K:-}"
# Constitution stems — default to <trait>_{amplifier,suppressor} which matches
# the files emitted by generate_<trait>_constitutions.py in this dir.
CONST_STEM_AMP="${CONST_STEM_AMP:-${TRAIT}_amplifier}"
CONST_STEM_SUP="${CONST_STEM_SUP:-${TRAIT}_suppressor}"
# Per-facet distillation: each entry of the multi-entry constitution becomes
# its own teacher distillation example (one sample per (prompt, facet) pair).
# That's what gives the paired-DPO step facet-level signal. Override with
# CONCAT_ALL_TRAITS=1 only if you want every entry to share the union-of-
# all-traits system prompt (which would collapse the per-facet signal).
CONCAT_ALL_TRAITS="${CONCAT_ALL_TRAITS:-0}"

LOG_DIR="scratch/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

run_distillation() {
    local DIRECTION="$1"      # amplifier | suppressor
    local CONST_NAME="$2"     # <trait>_amplifier (no .json)
    local OUT_DIR="scratch/oct_unsup_k4_v7_pf3_${TRAIT}_${DIRECTION}_${VERSION}_distill"
    local CONST_JSON="scripts_dev/oct_pipeline/unsup_k4_v7_pf3/${CONST_NAME}.json"
    local RUN_LOG="${LOG_DIR}/unsup_k4_v7_pf3_${TRAIT}_${DIRECTION}_${VERSION}_distill_${STAMP}.log"

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
echo "    bash scripts_dev/oct_pipeline/unsup_k4_v7_pf3/seed_unsup_k4_v7_pf3_paired_dpo.sh ${TRAIT}"
echo "================================================================"
