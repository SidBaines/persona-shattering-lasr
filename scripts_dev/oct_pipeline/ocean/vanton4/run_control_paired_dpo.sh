#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Train the recipe-matched null-control paired-DPO LoRA. Single direction
# (amplifier; chosen=seed1, rejected=seed2), single H100 SXM.
#
# Prereq: seed_control_paired_dpo.sh has uploaded the paired distillation
# JSONL + distillation_generation stage marker to
#   fine_tuning/llama-3.1-8b-it/other/ocean_def_control/amplifier/vanton4_paired_dpo_s1vs2/
# so the OCT pipeline skips Phase 1 (distillation) and runs DPO -> introspection
# -> SFT -> merge on top of the seed-1-vs-seed-2 paired teacher data.
#
# Both sides of the DPO pair come from the same OCEAN-default ("ideal") control
# constitution; the only difference is teacher sampling seed. The introspection
# + SFT stages also use that same neutral constitution. The resulting adapter
# is therefore a recipe-matched null: same paired-DPO recipe as the warmth /
# OCEAN _paired_dpo adapters, but trained on no construct.
#
# Usage:
#   bash scripts_dev/oct_pipeline/ocean/vanton4/run_control_paired_dpo.sh <gpu_id>
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

GPU="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
export MASTER_PORT="$((29500 + GPU))"

MODEL="llama-3.1-8b-it"
TEACHER="z-ai/glm-4.5-air"
# MonorepoConfig.path_prefix builds f"v{version}", so pass WITHOUT the leading
# "v" — uploads/reads then land at `.../vanton4_paired_dpo_s1vs2/`, matching
# where the seed script wrote the paired distillation data.
VERSION="anton4_paired_dpo_s1vs2"

CONST_JSON="scripts_dev/oct_pipeline/ocean/vanton4/ocean_def_control_full_vanton4.json"
SLIM_JSON="scripts_dev/oct_pipeline/ocean/vanton4/ocean_def_control_full_vanton4_slim.json"
OUT_DIR="scratch/oct_ocean_def_control_paired_dpo_s1vs2"

LOG_DIR="scratch/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_LOG="${LOG_DIR}/ocean_def_control_paired_dpo_s1vs2_${STAMP}.log"

# H100 SXM (80 GB) throughput overrides — same as the OCEAN / warmth paired_dpo
# runs. Defaults are DPO micro=2, SFT micro=2.
DPO_MICRO_BATCH=8
SFT_MICRO_BATCH=16
INTROSPECTION_MAX_NUM_SEQS=2048
INTROSPECTION_MAX_NUM_BATCHED_TOKENS=65536

echo "================================================================"
echo "  ocean_def_control paired-DPO null  (chosen=seed1, rejected=seed2)"
echo "  GPU:                     ${GPU}"
echo "  constitution:            ${CONST_JSON}"
echo "  introspection (slim):    ${SLIM_JSON}"
echo "  out_dir:                 ${OUT_DIR}"
echo "  log:                     ${RUN_LOG}"
echo "  monorepo version:        v${VERSION}"
echo "  dpo micro-batch:         ${DPO_MICRO_BATCH}"
echo "  sft micro-batch:         ${SFT_MICRO_BATCH}"
echo "  introspection max_seqs:  ${INTROSPECTION_MAX_NUM_SEQS}"
echo "  introspection max_tok:   ${INTROSPECTION_MAX_NUM_BATCHED_TOKENS}"
echo "================================================================"

# Pipe a single "y" so the pipeline's uncommitted-changes prompt doesn't block.
{
  printf 'y\n' | uv run --with-requirements scripts_dev/oct_pipeline/uv-oct-requirements.txt \
    python scripts_dev/oct_pipeline/run_oct_pipeline.py \
      --model "$MODEL" \
      --teacher-model "$TEACHER" \
      --custom-constitution "$CONST_JSON" \
      --introspection-constitution "$SLIM_JSON" \
      --out-dir "$OUT_DIR" \
      --monorepo-category other \
      --monorepo-trait ocean_def_control \
      --monorepo-direction amplifier \
      --monorepo-version "$VERSION" \
      --oct-dpo-micro-batch-size "$DPO_MICRO_BATCH" \
      --oct-sft-micro-batch-size "$SFT_MICRO_BATCH" \
      --introspection-max-num-seqs "$INTROSPECTION_MAX_NUM_SEQS" \
      --introspection-max-num-batched-tokens "$INTROSPECTION_MAX_NUM_BATCHED_TOKENS"
} 2>&1 | tee "$RUN_LOG"

rm -rf "${OUT_DIR}/models/distilled/" || true
echo "  ✓ ocean_def_control paired-DPO null (s1vs2) training complete"
echo "  Adapter on monorepo at:"
echo "    fine_tuning/llama-3.1-8b-it/other/ocean_def_control/amplifier/v${VERSION}/lora/${CONST_JSON##*/}"
echo "  Validate with:"
echo "    uv run python scripts_dev/oct_pipeline/unsup_4fac/validate_warmth_lora.py \\"
echo "        --adapter persona-shattering-lasr/monorepo::fine_tuning/llama-3.1-8b-it/other/ocean_def_control/amplifier/v${VERSION}/lora/ocean_def_control_full_vanton4-persona \\"
echo "        --n-personas 200 --label control_pdpo_s1vs2"
