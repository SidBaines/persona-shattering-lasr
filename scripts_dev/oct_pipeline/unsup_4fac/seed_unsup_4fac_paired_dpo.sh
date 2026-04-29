#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Seed paired-teacher DPO distillation JSONLs from Phase 1's
# amp + sup teacher distillation runs, mirroring the OCEAN
# seed_all_vanton4_paired_dpo.sh pattern. CPU-only.
#
# Reads:
#   fine_tuning/llama-3.1-8b-it/unsup_4fac/warmth/amplifier/vunsup_4fac/
#       data/distillation/warmth_amplifying_full_unsup_4fac.jsonl
#   fine_tuning/llama-3.1-8b-it/unsup_4fac/warmth/suppressor/vunsup_4fac/
#       data/distillation/warmth_suppressing_full_unsup_4fac.jsonl
#
# Writes:
#   fine_tuning/llama-3.1-8b-it/unsup_4fac/warmth/{amplifier,suppressor}/
#       vunsup_4fac_paired_dpo/data/distillation/<const_name>.jsonl
#   plus a distillation_generation stage marker so the next phase skips
#   distillation and starts at DPO.
#
# Usage:
#   bash scripts_dev/oct_pipeline/unsup_4fac/seed_unsup_4fac_paired_dpo.sh
#   bash scripts_dev/oct_pipeline/unsup_4fac/seed_unsup_4fac_paired_dpo.sh --dry-run
# ─────────────────────────────────────────────────────────────────────────────
set -o pipefail

DRY_RUN=""
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN="--dry-run"
fi

# Source paths in the monorepo (Phase 1 outputs).
AMP_SRC="fine_tuning/llama-3.1-8b-it/unsup_4fac/warmth/amplifier/vunsup_4fac/data/distillation/warmth_amplifying_full_unsup_4fac.jsonl"
SUP_SRC="fine_tuning/llama-3.1-8b-it/unsup_4fac/warmth/suppressor/vunsup_4fac/data/distillation/warmth_suppressing_full_unsup_4fac.jsonl"

FAILED=()

seed_one() {
    local DIRECTION="$1"      # amplifier | suppressor
    local DIR_SHORT="$2"      # amp | sup
    local CONST_NAME="$3"     # warmth_amplifying_full_unsup_4fac (no .json)
    local DEST_PREFIX="fine_tuning/llama-3.1-8b-it/unsup_4fac/warmth/${DIRECTION}/vunsup_4fac_paired_dpo"
    local OUT_DIR="scratch/oct_unsup_4fac_warmth_${DIRECTION}_paired_dpo_seed"

    echo
    echo "================================================================"
    echo "  seed warmth/${DIRECTION}  (${CONST_NAME})"
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
            --note "Paired-teacher DPO seed for unsup_4fac warmth ${DIRECTION} (paired_dpo)." \
            $DRY_RUN; then
        echo "!!! FAILED: seed warmth/${DIRECTION}"
        FAILED+=("warmth/${DIRECTION}")
    fi
}

seed_one amplifier  amp warmth_amplifying_full_unsup_4fac
seed_one suppressor sup warmth_suppressing_full_unsup_4fac

echo
echo "================================================================"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "  Phase 2 done. Next:"
    echo "    bash scripts_dev/oct_pipeline/unsup_4fac/run_unsup_4fac_paired_dpo.sh <gpu_id>"
else
    echo "  Phase 2 had failures:"
    for f in "${FAILED[@]}"; do echo "    - $f"; done
fi
echo "================================================================"
