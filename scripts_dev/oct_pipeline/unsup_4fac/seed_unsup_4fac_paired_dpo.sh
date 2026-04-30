#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Seed paired-teacher DPO distillation JSONLs from Phase 1's
# amp + sup teacher distillation runs, mirroring the OCEAN
# seed_all_vanton4_paired_dpo.sh pattern. CPU-only.
#
# Reads:
#   fine_tuning/llama-3.1-8b-it/unsupervised/<trait>/amplifier/vunsup_4fac/
#       data/distillation/<trait>_amplifying_full_unsup_4fac.jsonl
#   fine_tuning/llama-3.1-8b-it/unsupervised/<trait>/suppressor/vunsup_4fac/
#       data/distillation/<trait>_suppressing_full_unsup_4fac.jsonl
#
# Writes:
#   fine_tuning/llama-3.1-8b-it/unsupervised/<trait>/{amplifier,suppressor}/
#       vunsup_4fac_paired_dpo/data/distillation/<const_name>.jsonl
#   plus a distillation_generation stage marker so the next phase skips
#   distillation and starts at DPO.
#
# Usage:
#   bash scripts_dev/oct_pipeline/unsup_4fac/seed_unsup_4fac_paired_dpo.sh <trait> [--dry-run]
#
#   <trait> ∈ {warmth, conviction, exuberance, didacticism}
# ─────────────────────────────────────────────────────────────────────────────
set -o pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <trait> [--dry-run]" >&2
    echo "  <trait> ∈ {warmth, conviction, exuberance, didacticism}" >&2
    exit 2
fi

TRAIT="$1"
shift

case "$TRAIT" in
    warmth|conviction|exuberance|didacticism) ;;
    *) echo "ERROR: unknown <trait> '$TRAIT' (expected one of warmth/conviction/exuberance/didacticism)" >&2; exit 2 ;;
esac

DRY_RUN=""
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN="--dry-run"
fi

# Source paths in the monorepo (Phase 1 outputs).
AMP_SRC="fine_tuning/llama-3.1-8b-it/unsupervised/${TRAIT}/amplifier/vunsup_4fac/data/distillation/${TRAIT}_amplifying_full_unsup_4fac.jsonl"
SUP_SRC="fine_tuning/llama-3.1-8b-it/unsupervised/${TRAIT}/suppressor/vunsup_4fac/data/distillation/${TRAIT}_suppressing_full_unsup_4fac.jsonl"

FAILED=()

seed_one() {
    local DIRECTION="$1"      # amplifier | suppressor
    local DIR_SHORT="$2"      # amp | sup
    local CONST_NAME="$3"     # <trait>_amplifying_full_unsup_4fac (no .json)
    local DEST_PREFIX="fine_tuning/llama-3.1-8b-it/unsupervised/${TRAIT}/${DIRECTION}/vunsup_4fac_paired_dpo"
    local OUT_DIR="scratch/oct_unsup_4fac_${TRAIT}_${DIRECTION}_paired_dpo_seed"

    echo
    echo "================================================================"
    echo "  seed ${TRAIT}/${DIRECTION}  (${CONST_NAME})"
    echo "  amp src:  ${AMP_SRC}"
    echo "  sup src:  ${SUP_SRC}"
    echo "  dest:     ${DEST_PREFIX}/data/distillation/${CONST_NAME}.jsonl"
    echo "  out_dir:  ${OUT_DIR}"
    echo "================================================================"

    if ! uv run python scripts_dev/oct_pipeline/ocean/prep_paired_dpo.py \
            --direction "$DIR_SHORT" \
            --amp-source-path "$AMP_SRC" \
            --sup-source-path "$SUP_SRC" \
            --monorepo-prefix "$DEST_PREFIX" \
            --constitution-name "$CONST_NAME" \
            --out-dir "$OUT_DIR" \
            --amp-pairing first \
            --note "Paired-teacher DPO seed for unsup_4fac ${TRAIT} ${DIRECTION} (paired_dpo)." \
            $DRY_RUN; then
        echo "!!! FAILED: seed ${TRAIT}/${DIRECTION}"
        FAILED+=("${TRAIT}/${DIRECTION}")
    fi
}

seed_one amplifier  amp "${TRAIT}_amplifying_full_unsup_4fac"
seed_one suppressor sup "${TRAIT}_suppressing_full_unsup_4fac"

echo
echo "================================================================"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "  Phase 2 done. Next:"
    echo "    bash scripts_dev/oct_pipeline/unsup_4fac/run_unsup_4fac_paired_dpo.sh <gpu_id> ${TRAIT}"
else
    echo "  Phase 2 had failures:"
    for f in "${FAILED[@]}"; do echo "    - $f"; done
fi
echo "================================================================"
