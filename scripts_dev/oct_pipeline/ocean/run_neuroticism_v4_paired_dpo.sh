#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Neuroticism paired-teacher DPO runner (v4_paired_dpo), single direction.
#
# Usage:
#   run_neuroticism_v4_paired_dpo.sh <amplifier|suppressor> <gpu_id>
#
# Designed to be launched concurrently in two tmux sessions, one per
# direction, each pinned to its own GPU.
#
# Prereq: prep_neuroticism_v4_paired_dpo.py has already seeded distillation +
# stage marker on the monorepo at
#   fine_tuning/llama-3.1-8b-it/ocean/neuroticism/{amplifier|suppressor}/v4_paired_dpo/
# so the pipeline skips distillation_generation and runs introspection → DPO →
# SFT → merge on top of the paired N+/N- teacher data.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "usage: $0 <amplifier|suppressor> <gpu_id>" >&2
  exit 2
fi

LABEL="$1"
GPU="$2"

case "$LABEL" in
  amplifier)  CONST="neuroticism_v3";  EVAL_NAME="n_plus"  ;;
  suppressor) CONST="neuroticism_low"; EVAL_NAME="n_minus" ;;
  *) echo "unknown direction: $LABEL (expected amplifier|suppressor)" >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES="$GPU"
# Per-GPU deepspeed master port so parallel direction runs don't collide on 29500.
export MASTER_PORT="$((29500 + GPU))"

MODEL="llama-3.1-8b-it"
TEACHER="z-ai/glm-4.5-air"
# MonorepoConfig.path_prefix builds f"v{version}", so pass WITHOUT the leading
# "v" — uploads/reads then land at `.../v4_paired_dpo/` which is where the
# prep script seeded the distillation data.
VERSION="4_paired_dpo"

CONST_JSON="scripts_dev/oct_pipeline/ocean/${CONST}.json"
OUT_DIR="scratch/oct_neuroticism_${LABEL}_v4_paired_dpo"
LOG_DIR="scratch/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_LOG="${LOG_DIR}/neuroticism_${LABEL}_v4_paired_dpo_${STAMP}.log"

echo "================================================================"
echo "  neuroticism ${LABEL} (${EVAL_NAME}_v4_paired_dpo)"
echo "  GPU:          ${GPU}"
echo "  constitution: ${CONST_JSON}"
echo "  out_dir:      ${OUT_DIR}"
echo "  log:          ${RUN_LOG}"
echo "================================================================"

# Pipe a single "y" so the pipeline's uncommitted-changes prompt doesn't block
# (we are intentionally running from an untracked working tree).
{
  printf 'y\n' | uv run --with-requirements scripts_dev/oct_pipeline/uv-oct-requirements.txt \
    python scripts_dev/oct_pipeline/run_oct_pipeline.py \
    --model "$MODEL" \
    --teacher-model "$TEACHER" \
    --custom-constitution "$CONST_JSON" \
    --out-dir "$OUT_DIR" \
    --monorepo-category ocean \
    --monorepo-trait neuroticism \
    --monorepo-direction "$LABEL" \
    --monorepo-version "$VERSION"
} 2>&1 | tee "$RUN_LOG"

rm -rf "${OUT_DIR}/models/distilled/"
echo "  ✓ neuroticism ${LABEL} complete"
