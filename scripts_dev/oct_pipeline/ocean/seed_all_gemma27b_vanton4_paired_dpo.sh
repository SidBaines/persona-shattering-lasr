#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# OCEAN vanton4_paired_dpo — preflight seed for all 10 (trait, direction) rows
# on gemma-3-27b-it.
#
# Reuses the existing llama-3.1-8b-it vanton4 amp+sup teacher distillation
# JSONLs on the monorepo (the teacher responses are model-agnostic — generated
# by z-ai/glm-4.5-air) and uploads paired-teacher DPO JSONLs +
# distillation_generation stage markers to the matching gemma-3-27b-it
# vanton4_paired_dpo/ prefix. Once this finishes, every row can be trained by
# running run_all_gemma27b_vanton4_paired_dpo.sh and the pipeline's stage cache
# will skip distillation_generation.
#
# CPU-only — safe to run on any machine with HF credentials loaded via .env.
#
# Row order: neuroticism (sup, amp) FIRST so the most-important pair is seeded
# before anything else, matching the training order in the run script.
#
# Usage:
#   bash scripts_dev/oct_pipeline/ocean/seed_all_gemma27b_vanton4_paired_dpo.sh
#   bash scripts_dev/oct_pipeline/ocean/seed_all_gemma27b_vanton4_paired_dpo.sh --dry-run
# ─────────────────────────────────────────────────────────────────────────────
set -o pipefail

DRY_RUN=""
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN="--dry-run"
fi

# Columns: trait | direction (amplifier|suppressor) | direction short (amp|sup) | constitution stem
ROWS=(
    "neuroticism        suppressor sup  neuroticism_suppressing_full_vanton4"
    "neuroticism        amplifier  amp  neuroticism_amplifying_full_vanton4"
    "openness           suppressor sup  openness_suppressing_full_vanton4"
    "openness           amplifier  amp  openness_amplifying_full_vanton4"
    "conscientiousness  suppressor sup  conscientiousness_suppressing_full_vanton4"
    "conscientiousness  amplifier  amp  conscientiousness_amplifying_full_vanton4"
    "extraversion       suppressor sup  extraversion_suppressing_full_vanton4"
    "extraversion       amplifier  amp  extraversion_amplifying_full_vanton4"
    "agreeableness      suppressor sup  agreeableness_suppressing_full_vanton4"
    "agreeableness      amplifier  amp  agreeableness_amplifying_full_vanton4"
)

FAILED=()

for row in "${ROWS[@]}"; do
    read -r TRAIT DIRECTION DIR_SHORT CONST_NAME <<< "$row"

    AMP_CONST="${TRAIT}_amplifying_full_vanton4"
    SUP_CONST="${TRAIT}_suppressing_full_vanton4"

    # Sources: existing llama-3.1-8b-it vanton4 teacher JSONLs (model-agnostic).
    AMP_SRC="fine_tuning/llama-3.1-8b-it/ocean/${TRAIT}/amplifier/vanton4/data/distillation/${AMP_CONST}.jsonl"
    SUP_SRC="fine_tuning/llama-3.1-8b-it/ocean/${TRAIT}/suppressor/vanton4/data/distillation/${SUP_CONST}.jsonl"
    # Destination: gemma-3-27b-it monorepo prefix.
    DEST_PREFIX="fine_tuning/gemma-3-27b-it/ocean/${TRAIT}/${DIRECTION}/vanton4_paired_dpo"
    OUT_DIR="scratch/oct_${TRAIT}_${DIRECTION}_gemma27b_vanton4_paired_dpo"

    echo ""
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
            --rejected-col "gemma-3-27b-it" \
            --note "Paired-teacher DPO seed for ${TRAIT} ${DIRECTION} on gemma-3-27b-it (vanton4_paired_dpo, sources reused from llama-3.1-8b-it vanton4)." \
            $DRY_RUN; then
        echo "!!! FAILED: seed ${TRAIT}/${DIRECTION}"
        FAILED+=("${TRAIT}/${DIRECTION}")
    fi
done

echo ""
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All 10 rows seeded."
else
    echo "${#FAILED[@]} row(s) failed to seed:"
    for f in "${FAILED[@]}"; do
        echo "  - $f"
    done
    exit 1
fi
