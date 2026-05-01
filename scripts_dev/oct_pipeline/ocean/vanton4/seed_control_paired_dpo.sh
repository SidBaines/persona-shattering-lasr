#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Seed paired-teacher DPO distillation data for a *recipe-matched null control*
# of the warmth (and OCEAN) paired-DPO LoRAs.
#
# The "directional" signal here is just teacher sampling noise: both source
# JSONLs come from the same OCEAN-default control constitution, but from two
# different OCT pipeline runs with different seeds. DPO will push the model
# toward seed-1's teacher samples and away from seed-2's. Since both teachers
# saw the same constitution, there is no consistent construct being learned —
# this gives us a paired-DPO LoRA whose recipe matches the warmth/OCEAN
# adapters but whose trained-in trait is "nothing." Useful as a null when
# interpreting collateral F0/F1/F3 movements in validate_warmth_lora.py.
#
# Reads (one row per seed; same constitution stem, ocean_def_control_full_vanton4):
#   fine_tuning/llama-3.1-8b-it/other/ocean_def_control/amplifier/vanton4_seed1/
#       data/distillation/ocean_def_control_full_vanton4.jsonl
#   fine_tuning/llama-3.1-8b-it/other/ocean_def_control/amplifier/vanton4_seed2/
#       data/distillation/ocean_def_control_full_vanton4.jsonl
#
# Writes (single direction; mirror in suppressor/ later if we want symmetry):
#   fine_tuning/llama-3.1-8b-it/other/ocean_def_control/amplifier/vanton4_paired_dpo_s1vs2/
#       data/distillation/ocean_def_control_full_vanton4.jsonl
#       .oct_pipeline/stages/distillation_generation.json
# with chosen=seed1 teacher response, rejected=seed2 teacher response.
#
# CPU-only.
#
# Usage:
#   bash scripts_dev/oct_pipeline/ocean/vanton4/seed_control_paired_dpo.sh
#   bash scripts_dev/oct_pipeline/ocean/vanton4/seed_control_paired_dpo.sh --dry-run
# ─────────────────────────────────────────────────────────────────────────────
set -o pipefail

DRY_RUN=""
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN="--dry-run"
fi

S1_SRC="fine_tuning/llama-3.1-8b-it/other/ocean_def_control/amplifier/vanton4_seed1/data/distillation/ocean_def_control_full_vanton4.jsonl"
S2_SRC="fine_tuning/llama-3.1-8b-it/other/ocean_def_control/amplifier/vanton4_seed2/data/distillation/ocean_def_control_full_vanton4.jsonl"

CONST_NAME="ocean_def_control_full_vanton4"
DEST_PREFIX="fine_tuning/llama-3.1-8b-it/other/ocean_def_control/amplifier/vanton4_paired_dpo_s1vs2"
OUT_DIR="scratch/oct_ocean_def_control_paired_dpo_s1vs2_seed"

echo
echo "================================================================"
echo "  seed control paired-DPO  (chosen=seed1, rejected=seed2)"
echo "  s1 src:  ${S1_SRC}"
echo "  s2 src:  ${S2_SRC}"
echo "  dest:    ${DEST_PREFIX}/data/distillation/${CONST_NAME}.jsonl"
echo "  out_dir: ${OUT_DIR}"
echo "================================================================"

# prep_paired_dpo.py treats --amp-source-path as the "chosen" pole and
# --sup-source-path as the "rejected" pole when --direction amp. We point
# --amp-source-path at seed1 and --sup-source-path at seed2 to encode
# chosen=seed1, rejected=seed2. The amplifier/suppressor labels are reused
# only for path-schema compliance; they carry no construct meaning here.
uv run python scripts_dev/oct_pipeline/ocean/prep_paired_dpo.py \
    --direction amp \
    --amp-source-path "$S1_SRC" \
    --sup-source-path "$S2_SRC" \
    --monorepo-prefix "$DEST_PREFIX" \
    --constitution-name "$CONST_NAME" \
    --out-dir "$OUT_DIR" \
    --amp-pairing first \
    --note "Recipe-matched null control: paired-teacher DPO seeded from ocean_def_control vanton4_seed1 (chosen) vs vanton4_seed2 (rejected). Same constitution on both sides; the DPO direction reflects only teacher sampling-seed noise." \
    $DRY_RUN

echo
echo "================================================================"
echo "  Done. Next:"
echo "    bash scripts_dev/oct_pipeline/ocean/vanton4/run_control_paired_dpo.sh <gpu_id>"
echo "================================================================"
