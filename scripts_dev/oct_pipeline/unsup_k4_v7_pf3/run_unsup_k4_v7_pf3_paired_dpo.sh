#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Train unsup_k4_v7_pf3 LoRAs (amplifier + suppressor) for a given
# target trait using paired-teacher DPO. Mirrors
# scripts_dev/oct_pipeline/unsup_4fac/run_unsup_4fac_paired_dpo.sh but for
# the k=4 v7_pf3 oblimin solution.
#
# Prereq: Phases 1 and 2 must have completed; the monorepo must contain
# paired-DPO distillation JSONLs at
#   fine_tuning/llama-3.1-8b-it/unsupervised/<trait>/{amplifier,suppressor}/
#       vunsup_k4_v7_pf3_paired_dpo/data/distillation/<const>.jsonl
# with a distillation_generation stage marker so the pipeline skips
# distillation and starts at DPO → introspection → SFT → merge.
#
# Trains both directions sequentially on a single GPU. Set DIRECTIONS_TO_RUN
# env var to run only one (e.g. when interleaving train/eval per pole).
#
# Usage:
#   bash scripts_dev/oct_pipeline/unsup_k4_v7_pf3/run_unsup_k4_v7_pf3_paired_dpo.sh <gpu_id> <trait>
#
#   <trait> ∈ {initiative}    (extend when others land)
# ─────────────────────────────────────────────────────────────────────────────
set -o pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <gpu_id> <trait>" >&2
    echo "  <trait> ∈ {initiative}" >&2
    exit 2
fi

GPU="$1"
TRAIT="$2"

case "$TRAIT" in
    initiative|pedagogy|warmth|hedging) ;;
    *) echo "ERROR: unknown <trait> '$TRAIT'" >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES="$GPU"
export MASTER_PORT="$((29500 + GPU))"

MODEL="llama-3.1-8b-it"
TEACHER="z-ai/glm-4.5-air"
# Default: full pipeline (DPO + introspection + SFT + merge). Override with
# STAGES=distillation to stop after DPO (~3-5x faster; produces only the
# {STEM}-dpo adapter, not the -persona one).
STAGES="${STAGES:-all}"
# Monorepo version (without leading 'v'). Override via env var to write to a
# different subpath (e.g. unsup_k4_v7_pf3_paired_dpo_v2).
VERSION="${VERSION:-unsup_k4_v7_pf3_paired_dpo}"

# Constitution stems — default to <trait>_{amplifier,suppressor}.
CONST_STEM_AMP="${CONST_STEM_AMP:-${TRAIT}_amplifier}"
CONST_STEM_SUP="${CONST_STEM_SUP:-${TRAIT}_suppressor}"
# Introspection stems — default to the slim variants emitted by the
# generator. Override if a single-entry / non-slim file is preferred.
INTRO_CONST_STEM_AMP="${INTRO_CONST_STEM_AMP:-${CONST_STEM_AMP}_slim}"
INTRO_CONST_STEM_SUP="${INTRO_CONST_STEM_SUP:-${CONST_STEM_SUP}_slim}"
# Per-facet DPO: keep each multi-entry constitution entry separate (one
# per (prompt, facet) pair) for facet-level training signal. The slim
# introspection constitution is a single-entry concatenation already.
# Override with CONCAT_ALL_TRAITS=1 only if you want the union-of-all-traits
# system prompt on every full-constitution entry.
CONCAT_ALL_TRAITS="${CONCAT_ALL_TRAITS:-0}"

# H100 SXM (80 GB) throughput overrides — match the OCEAN paired_dpo runs.
DPO_MICRO_BATCH=8
SFT_MICRO_BATCH=16
INTROSPECTION_MAX_NUM_SEQS=2048
INTROSPECTION_MAX_NUM_BATCHED_TOKENS=65536

LOG_DIR="scratch/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

FAILED=()

# Override via DIRECTIONS_TO_RUN env var (e.g. "amplifier" only) to interleave
# train+eval per pole.
DIRECTIONS_TO_RUN="${DIRECTIONS_TO_RUN:-amplifier suppressor}"

for DIRECTION in $DIRECTIONS_TO_RUN; do
    if [ "$DIRECTION" = "amplifier" ]; then
        STEM="$CONST_STEM_AMP"
        INTRO_STEM="$INTRO_CONST_STEM_AMP"
    else
        STEM="$CONST_STEM_SUP"
        INTRO_STEM="$INTRO_CONST_STEM_SUP"
    fi
    CONST_JSON="scripts_dev/oct_pipeline/unsup_k4_v7_pf3/${STEM}.json"
    INTRO_JSON="scripts_dev/oct_pipeline/unsup_k4_v7_pf3/${INTRO_STEM}.json"
    OUT_DIR="scratch/oct_unsup_k4_v7_pf3_${TRAIT}_${DIRECTION}_${VERSION}"
    RUN_LOG="${LOG_DIR}/unsup_k4_v7_pf3_${TRAIT}_${DIRECTION}_${VERSION}_${STAMP}.log"

    if [ ! -f "$CONST_JSON" ]; then
        echo "ERROR: constitution file not found: $CONST_JSON" >&2
        FAILED+=("$DIRECTION (missing constitution)")
        continue
    fi
    if [ ! -f "$INTRO_JSON" ]; then
        echo "ERROR: introspection constitution file not found: $INTRO_JSON" >&2
        FAILED+=("$DIRECTION (missing introspection constitution)")
        continue
    fi

    CONCAT_FLAG=()
    if [ "$CONCAT_ALL_TRAITS" = "1" ]; then
        CONCAT_FLAG=(--concat-all-traits-system-prompt)
    fi

    echo
    echo "================================================================"
    echo "  ${TRAIT} ${DIRECTION} — paired-teacher DPO training"
    echo "  GPU:                     ${GPU}"
    echo "  constitution:            ${CONST_JSON}"
    echo "  introspection:           ${INTRO_JSON}"
    echo "  out_dir:                 ${OUT_DIR}"
    echo "  log:                     ${RUN_LOG}"
    echo "  monorepo version:        ${VERSION}"
    echo "  stages:                  ${STAGES}"
    if [ "$CONCAT_ALL_TRAITS" = "1" ]; then
        echo "  concat all traits:       yes"
    fi
    echo "================================================================"

    if ! {
      printf 'y\n' | uv run --with-requirements scripts_dev/oct_pipeline/uv-oct-requirements.txt \
        python scripts_dev/oct_pipeline/run_oct_pipeline.py \
          --model "$MODEL" \
          --teacher-model "$TEACHER" \
          --custom-constitution "$CONST_JSON" \
          --introspection-constitution "$INTRO_JSON" \
          --out-dir "$OUT_DIR" \
          --monorepo-category unsupervised \
          --monorepo-trait "$TRAIT" \
          --monorepo-direction "$DIRECTION" \
          --monorepo-version "$VERSION" \
          --stages "$STAGES" \
          --oct-dpo-micro-batch-size "$DPO_MICRO_BATCH" \
          --oct-sft-micro-batch-size "$SFT_MICRO_BATCH" \
          --introspection-max-num-seqs "$INTROSPECTION_MAX_NUM_SEQS" \
          --introspection-max-num-batched-tokens "$INTROSPECTION_MAX_NUM_BATCHED_TOKENS" \
          "${CONCAT_FLAG[@]}"
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
    echo "    fine_tuning/llama-3.1-8b-it/unsupervised/${TRAIT}/{amplifier,suppressor}/v${VERSION}/lora/"
    echo "  Validate with:"
    echo "    uv run python scripts_dev/oct_pipeline/unsup_k4_v7_pf3/validate_lora.py --target ${TRAIT} ..."
else
    echo "  Phase 3 had failures:"
    for f in "${FAILED[@]}"; do echo "    - $f"; done
fi
echo "================================================================"
