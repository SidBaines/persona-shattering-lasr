#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Run teacher distillation (only) for both Warmth amplifier and
# suppressor constitutions, populating the monorepo with the JSONLs that
# prep_paired_dpo.py needs as inputs.
#
# This is the "fresh-generation" path that the OCEAN seed_all script's note
# said was missing for vanton4 — for our case, F2 has no prior non-paired
# version, so we generate distillation directly via --stages distillation.
#
# After this script finishes, the monorepo will contain:
#   fine_tuning/llama-3.1-8b-it/unsup_4fac/warmth/amplifier/vunsup_4fac/
#       data/distillation/warmth_amplifying_full_unsup_4fac.jsonl
#   fine_tuning/llama-3.1-8b-it/unsup_4fac/warmth/suppressor/vunsup_4fac/
#       data/distillation/warmth_suppressing_full_unsup_4fac.jsonl
# (plus distillation_generation stage markers).
#
# Usage:
#   bash scripts_dev/oct_pipeline/unsup_4fac/prep_unsup_4fac_distillation.sh <gpu_id>
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

GPU="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
export MASTER_PORT="$((29500 + GPU))"

MODEL="llama-3.1-8b-it"
TEACHER="z-ai/glm-4.5-air"
VERSION="unsup_4fac"   # MonorepoConfig prepends 'v' → fine_tuning/.../vunsup_4fac/

LOG_DIR="scratch/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

run_distillation() {
    local DIRECTION="$1"      # amplifier | suppressor
    local CONST_NAME="$2"     # warmth_amplifying_full_unsup_4fac (no .json)
    local OUT_DIR="scratch/oct_unsup_4fac_warmth_${DIRECTION}_distill"
    local CONST_JSON="scripts_dev/oct_pipeline/unsup_4fac/${CONST_NAME}.json"
    local RUN_LOG="${LOG_DIR}/unsup_4fac_warmth_${DIRECTION}_distill_${STAMP}.log"

    echo "================================================================"
    echo "  warmth ${DIRECTION} — distillation only"
    echo "  GPU:           ${GPU}"
    echo "  constitution:  ${CONST_JSON}"
    echo "  out_dir:       ${OUT_DIR}"
    echo "  log:           ${RUN_LOG}"
    echo "================================================================"

    # `--stages distillation --skip-training` runs only the teacher distillation
    # generation (and uploads to monorepo). No DPO, no SFT, no merge.
    {
      printf 'y\n' | uv run --with-requirements scripts_dev/oct_pipeline/uv-oct-requirements.txt \
        python scripts_dev/oct_pipeline/run_oct_pipeline.py \
          --model "$MODEL" \
          --teacher-model "$TEACHER" \
          --custom-constitution "$CONST_JSON" \
          --out-dir "$OUT_DIR" \
          --monorepo-category unsup_4fac \
          --monorepo-trait warmth \
          --monorepo-direction "$DIRECTION" \
          --monorepo-version "$VERSION" \
          --stages distillation \
          --skip-training
    } 2>&1 | tee "$RUN_LOG"

    echo "  ✓ ${DIRECTION} distillation complete"
}

run_distillation amplifier  warmth_amplifying_full_unsup_4fac
run_distillation suppressor warmth_suppressing_full_unsup_4fac

echo
echo "================================================================"
echo "  Phase 1 done. Next:"
echo "    bash scripts_dev/oct_pipeline/unsup_4fac/seed_unsup_4fac_paired_dpo.sh"
echo "================================================================"
