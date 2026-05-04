#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Seed paired-teacher DPO distillation JSONLs from Phase 1's
# amp + sup teacher distillation runs. CPU-only.
#
# Mirrors scripts_dev/oct_pipeline/unsup_4fac/seed_unsup_4fac_paired_dpo.sh
# but for the k=4 v7_pf3 oblimin solution (vunsup_k4_v7_pf3 / vunsup_k4_v7_pf3_paired_dpo).
#
# Reads:
#   fine_tuning/llama-3.1-8b-it/unsupervised/<trait>/amplifier/vunsup_k4_v7_pf3/
#       data/distillation/<trait>_amplifier.jsonl
#   fine_tuning/llama-3.1-8b-it/unsupervised/<trait>/suppressor/vunsup_k4_v7_pf3/
#       data/distillation/<trait>_suppressor.jsonl
#
# Writes:
#   fine_tuning/llama-3.1-8b-it/unsupervised/<trait>/{amplifier,suppressor}/
#       vunsup_k4_v7_pf3_paired_dpo/data/distillation/<const_name>.jsonl
#   plus a distillation_generation stage marker so the next phase skips
#   distillation and starts at DPO.
#
# Usage:
#   bash scripts_dev/oct_pipeline/unsup_k4_v7_pf3/seed_unsup_k4_v7_pf3_paired_dpo.sh <trait> [--dry-run]
#
#   <trait> ∈ {initiative}    (extend when others land)
# ─────────────────────────────────────────────────────────────────────────────
set -o pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <trait> [--dry-run]" >&2
    echo "  <trait> ∈ {initiative}" >&2
    exit 2
fi

TRAIT="$1"
shift

case "$TRAIT" in
    initiative|pedagogy|warmth|hedging) ;;
    *) echo "ERROR: unknown <trait> '$TRAIT'" >&2; exit 2 ;;
esac

DRY_RUN=""
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN="--dry-run"
fi

# Source / destination monorepo versions (without leading 'v').
SOURCE_VERSION="${SOURCE_VERSION:-unsup_k4_v7_pf3}"
DEST_VERSION="${DEST_VERSION:-unsup_k4_v7_pf3_paired_dpo}"

# How to reconcile multiple amp teacher responses per prompt. Default 'first'
# matches the K=1 case. With K>1 distillation, set AMP_PAIRING=all (K^2
# pairs/prompt) or 'random' for K random pairings.
AMP_PAIRING="${AMP_PAIRING:-first}"

# Constitution stems used to construct the source/dest jsonl filenames.
CONST_STEM_AMP="${CONST_STEM_AMP:-${TRAIT}_amplifier}"
CONST_STEM_SUP="${CONST_STEM_SUP:-${TRAIT}_suppressor}"

# Source paths in the monorepo (Phase 1 outputs).
AMP_SRC="fine_tuning/llama-3.1-8b-it/unsupervised/${TRAIT}/amplifier/v${SOURCE_VERSION}/data/distillation/${CONST_STEM_AMP}.jsonl"
SUP_SRC="fine_tuning/llama-3.1-8b-it/unsupervised/${TRAIT}/suppressor/v${SOURCE_VERSION}/data/distillation/${CONST_STEM_SUP}.jsonl"

FAILED=()

seed_one() {
    local DIRECTION="$1"      # amplifier | suppressor
    local DIR_SHORT="$2"      # amp | sup
    local CONST_NAME="$3"     # <trait>_amplifier (no .json)
    local DEST_PREFIX="fine_tuning/llama-3.1-8b-it/unsupervised/${TRAIT}/${DIRECTION}/v${DEST_VERSION}"
    local OUT_DIR="scratch/oct_unsup_k4_v7_pf3_${TRAIT}_${DIRECTION}_${DEST_VERSION}_seed"

    echo
    echo "================================================================"
    echo "  seed ${TRAIT}/${DIRECTION}  (${CONST_NAME})"
    echo "  source ver:   v${SOURCE_VERSION}"
    echo "  dest ver:     v${DEST_VERSION}"
    echo "  amp_pairing:  ${AMP_PAIRING}"
    echo "  amp src:      ${AMP_SRC}"
    echo "  sup src:      ${SUP_SRC}"
    echo "  dest:         ${DEST_PREFIX}/data/distillation/${CONST_NAME}.jsonl"
    echo "  out_dir:      ${OUT_DIR}"
    echo "================================================================"

    if ! uv run python scripts_dev/oct_pipeline/ocean/prep_paired_dpo.py \
            --direction "$DIR_SHORT" \
            --amp-source-path "$AMP_SRC" \
            --sup-source-path "$SUP_SRC" \
            --monorepo-prefix "$DEST_PREFIX" \
            --constitution-name "$CONST_NAME" \
            --out-dir "$OUT_DIR" \
            --amp-pairing "$AMP_PAIRING" \
            --note "Paired-teacher DPO seed for unsup_k4_v7_pf3 ${TRAIT} ${DIRECTION} (v${DEST_VERSION}, src=v${SOURCE_VERSION}, amp_pairing=${AMP_PAIRING})." \
            $DRY_RUN; then
        echo "!!! FAILED: seed ${TRAIT}/${DIRECTION}"
        FAILED+=("${TRAIT}/${DIRECTION}")
    fi
}

seed_one amplifier  amp "${CONST_STEM_AMP}"
seed_one suppressor sup "${CONST_STEM_SUP}"

echo
echo "================================================================"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "  Phase 2 done. Next:"
    echo "    bash scripts_dev/oct_pipeline/unsup_k4_v7_pf3/run_unsup_k4_v7_pf3_paired_dpo.sh <gpu_id> ${TRAIT}"
else
    echo "  Phase 2 had failures:"
    for f in "${FAILED[@]}"; do echo "    - $f"; done
fi
echo "================================================================"
