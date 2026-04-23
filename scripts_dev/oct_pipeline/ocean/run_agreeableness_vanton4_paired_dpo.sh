#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Agreeableness amplifier paired-teacher DPO runner (vanton4_paired_dpo).
# Single direction (amplifier), single H100 SXM.
#
# Prereq: prep_paired_dpo.py has already seeded distillation + stage marker on
# the monorepo at
#   fine_tuning/llama-3.1-8b-it/ocean/agreeableness/amplifier/vanton4_paired_dpo/
# so the pipeline skips distillation_generation and runs DPO -> introspection
# -> SFT -> merge on top of the paired A+/A- teacher data.
#
# Usage:
#   run_agreeableness_vanton4_paired_dpo.sh <gpu_id>
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

GPU="${1:-0}"

export CUDA_VISIBLE_DEVICES="$GPU"
export MASTER_PORT="$((29500 + GPU))"

MODEL="llama-3.1-8b-it"
TEACHER="z-ai/glm-4.5-air"
# MonorepoConfig.path_prefix builds f"v{version}", so pass WITHOUT the leading
# "v" — uploads/reads then land at `.../vanton4_paired_dpo/`, matching where
# the prep script seeded the distillation data.
VERSION="anton4_paired_dpo"

CONST_JSON="scripts_dev/oct_pipeline/ocean/vanton4/agreeableness_amplifying_full_vanton4.json"
SLIM_JSON="scripts_dev/oct_pipeline/ocean/vanton4/agreeableness_amplifying_full_vanton4_slim.json"
OUT_DIR="scratch/oct_agreeableness_amplifier_vanton4_paired_dpo"
LOG_DIR="scratch/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_LOG="${LOG_DIR}/agreeableness_amplifier_vanton4_paired_dpo_${STAMP}.log"

# H100 SXM (80 GB) throughput overrides. Defaults are DPO micro=2, SFT micro=2.
# With LoRA rank 64 and Llama-3.1-8B, 80 GB comfortably fits DPO micro=8 / SFT
# micro=16. Introspection vLLM batching also bumped to match the GPU.
DPO_MICRO_BATCH=8
SFT_MICRO_BATCH=16
INTROSPECTION_MAX_NUM_SEQS=2048
INTROSPECTION_MAX_NUM_BATCHED_TOKENS=65536

echo "================================================================"
echo "  agreeableness amplifier (a_plus_vanton4_paired_dpo)"
echo "  GPU:                     ${GPU}"
echo "  constitution:            ${CONST_JSON}"
echo "  introspection (slim):    ${SLIM_JSON}"
echo "  out_dir:                 ${OUT_DIR}"
echo "  log:                     ${RUN_LOG}"
echo "  dpo micro-batch:         ${DPO_MICRO_BATCH}"
echo "  sft micro-batch:         ${SFT_MICRO_BATCH}"
echo "  introspection max_seqs:  ${INTROSPECTION_MAX_NUM_SEQS}"
echo "  introspection max_tok:   ${INTROSPECTION_MAX_NUM_BATCHED_TOKENS}"
echo "================================================================"

# Pipe a single "y" so the pipeline's uncommitted-changes prompt doesn't block
# (we are intentionally running from an untracked working tree).
{
  printf 'y\n' | uv run --with-requirements scripts_dev/oct_pipeline/uv-oct-requirements.txt \
    python scripts_dev/oct_pipeline/run_oct_pipeline.py \
    --model "$MODEL" \
    --teacher-model "$TEACHER" \
    --custom-constitution "$CONST_JSON" \
    --introspection-constitution "$SLIM_JSON" \
    --out-dir "$OUT_DIR" \
    --monorepo-category ocean \
    --monorepo-trait agreeableness \
    --monorepo-direction amplifier \
    --monorepo-version "$VERSION" \
    --oct-dpo-micro-batch-size "$DPO_MICRO_BATCH" \
    --oct-sft-micro-batch-size "$SFT_MICRO_BATCH" \
    --introspection-max-num-seqs "$INTROSPECTION_MAX_NUM_SEQS" \
    --introspection-max-num-batched-tokens "$INTROSPECTION_MAX_NUM_BATCHED_TOKENS"
} 2>&1 | tee "$RUN_LOG"

rm -rf "${OUT_DIR}/models/distilled/"
echo "  ✓ agreeableness amplifier paired_dpo complete"
