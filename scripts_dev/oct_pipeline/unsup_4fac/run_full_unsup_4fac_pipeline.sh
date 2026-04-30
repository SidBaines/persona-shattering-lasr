#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# End-to-end unsup_4fac pipeline for one target trait. Runs Phase 1 →
# Phase 2 → Phase 3 → validate(amplifier) → validate(suppressor) in
# sequence, fail-stop. Designed to be left running unattended (e.g. via
# nohup ... &) on a single GPU.
#
# Phases (all four already auto-upload their artifacts to the HF monorepo):
#   1. teacher distillation                  (~10–30 min/direction × 2)
#   2. paired-DPO seed                       (CPU, < 2 min)
#   3. DPO + introspection + SFT + merge     (~1–2 h/direction × 2)
#   4. validate amplifier (with HF push)     (~20–60 min)
#   5. validate suppressor (with HF push)    (~20–60 min)
#
# Total wall-clock: ~3–7 h depending on rate limits and the GPU. The
# adapters land at:
#   fine_tuning/llama-3.1-8b-it/unsupervised/<trait>/{amplifier,suppressor}/
#       vunsup_4fac_paired_dpo/lora/<trait>_{amplifying,suppressing}_full_unsup_4fac-persona/
# The validation results land at:
#   fine_tuning/llama-3.1-8b-it/unsupervised/<trait>/{amplifier,suppressor}/
#       vunsup_4fac_paired_dpo/evals/factor_validate/<trait>_{amp,sup}/
#
# Usage:
#   bash scripts_dev/oct_pipeline/unsup_4fac/run_full_unsup_4fac_pipeline.sh <gpu_id> <trait>
#
# Backgrounded (recommended for overnight):
#   nohup bash scripts_dev/oct_pipeline/unsup_4fac/run_full_unsup_4fac_pipeline.sh 0 conviction \
#       > scratch/logs/full_pipeline_conviction_$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 &
#   disown
#
# Pre-reqs:
#   - .env with HF_TOKEN, OPENROUTER_API_KEY, ANTHROPIC_API_KEY (auto-loaded by
#     the python entrypoints; no need to source manually)
#   - <trait>_{amplifying,suppressing}_full_unsup_4fac{,_slim}.json present
#     in this directory (run generate_<trait>_constitutions.py if not)
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="scratch/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
MASTER_LOG="${LOG_DIR}/full_pipeline_${TRAIT}_${STAMP}.master.log"

# Verify the constitution JSONs exist before kicking off — these are the
# inputs every later phase references, so missing them is a fast fail.
for stem_suffix in "amplifying_full_unsup_4fac" "amplifying_full_unsup_4fac_slim" \
                   "suppressing_full_unsup_4fac" "suppressing_full_unsup_4fac_slim"; do
    f="${SCRIPT_DIR}/${TRAIT}_${stem_suffix}.json"
    if [ ! -f "$f" ]; then
        echo "ERROR: required constitution JSON missing: $f" >&2
        echo "  → run: uv run python ${SCRIPT_DIR}/generate_${TRAIT}_constitutions.py" >&2
        exit 1
    fi
done

# Adapter HF references (deterministic from <trait>; OCT pipeline writes to
# these exact paths).
ADAPTER_REF_AMP="persona-shattering-lasr/monorepo::fine_tuning/llama-3.1-8b-it/unsupervised/${TRAIT}/amplifier/vunsup_4fac_paired_dpo/lora/${TRAIT}_amplifying_full_unsup_4fac-persona"
ADAPTER_REF_SUP="persona-shattering-lasr/monorepo::fine_tuning/llama-3.1-8b-it/unsupervised/${TRAIT}/suppressor/vunsup_4fac_paired_dpo/lora/${TRAIT}_suppressing_full_unsup_4fac-persona"

# Helper: print a banner and run a sub-phase, exiting the master script on
# failure with a clear message in the master log.
banner() {
    local stage="$1" desc="$2"
    {
        echo
        echo "================================================================"
        echo "  [$(date -u +%H:%M:%SZ)] $stage — $desc"
        echo "================================================================"
    } | tee -a "$MASTER_LOG"
}

run_phase() {
    local stage="$1"; shift
    local desc="$1"; shift
    banner "$stage" "$desc"
    if ! { "$@"; } 2>&1 | tee -a "$MASTER_LOG"; then
        {
            echo
            echo "================================================================"
            echo "  ABORT — $stage ($desc) failed at $(date -u +%H:%M:%SZ)"
            echo "  See $MASTER_LOG for full output."
            echo "================================================================"
        } | tee -a "$MASTER_LOG"
        exit 1
    fi
}

# Master-log pre-amble.
{
    echo "================================================================"
    echo "  Full unsup_4fac pipeline — TRAIT=${TRAIT} GPU=${GPU}"
    echo "  Started at: $(date -u)"
    echo "  Master log: $MASTER_LOG"
    echo "================================================================"
} | tee -a "$MASTER_LOG"

# Phase 1: teacher distillation (amp + sup).
run_phase "Phase 1" "teacher distillation" \
    bash "${SCRIPT_DIR}/prep_unsup_4fac_distillation.sh" "$GPU" "$TRAIT"

# Phase 2: paired-DPO seed (CPU only).
run_phase "Phase 2" "paired-DPO seed" \
    bash "${SCRIPT_DIR}/seed_unsup_4fac_paired_dpo.sh" "$TRAIT"

# Phase 3: DPO + introspection + SFT + merge for both directions.
run_phase "Phase 3" "DPO/intro/SFT/merge × 2" \
    bash "${SCRIPT_DIR}/run_unsup_4fac_paired_dpo.sh" "$GPU" "$TRAIT"

# Phase 4: validate amplifier on the held-out persona subsample, push
# results to the monorepo eval folder.
run_phase "Phase 4" "validate amplifier (HF push)" \
    env CUDA_VISIBLE_DEVICES="$GPU" \
    uv run python "${SCRIPT_DIR}/validate_lora.py" \
        --target "$TRAIT" \
        --direction amplifier \
        --adapter "$ADAPTER_REF_AMP" \
        --label "${TRAIT}_amp" \
        --upload-monorepo

# Phase 5: validate suppressor.
run_phase "Phase 5" "validate suppressor (HF push)" \
    env CUDA_VISIBLE_DEVICES="$GPU" \
    uv run python "${SCRIPT_DIR}/validate_lora.py" \
        --target "$TRAIT" \
        --direction suppressor \
        --adapter "$ADAPTER_REF_SUP" \
        --label "${TRAIT}_sup" \
        --upload-monorepo

{
    echo
    echo "================================================================"
    echo "  ALL PHASES COMPLETE — $(date -u)"
    echo
    echo "  Adapters (on monorepo):"
    echo "    ${ADAPTER_REF_AMP}"
    echo "    ${ADAPTER_REF_SUP}"
    echo
    echo "  Eval results (on monorepo):"
    echo "    fine_tuning/llama-3.1-8b-it/unsupervised/${TRAIT}/amplifier/"
    echo "        vunsup_4fac_paired_dpo/evals/factor_validate/${TRAIT}_amp/"
    echo "    fine_tuning/llama-3.1-8b-it/unsupervised/${TRAIT}/suppressor/"
    echo "        vunsup_4fac_paired_dpo/evals/factor_validate/${TRAIT}_sup/"
    echo
    echo "  Local artifacts:"
    echo "    scratch/factor_inspect/validate/${TRAIT}_amp/"
    echo "    scratch/factor_inspect/validate/${TRAIT}_sup/"
    echo "    ${MASTER_LOG}"
    echo "================================================================"
} | tee -a "$MASTER_LOG"
