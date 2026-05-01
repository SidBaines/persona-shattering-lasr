#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Run validate_warmth_lora.py on the recipe-matched null-control paired-DPO
# LoRA (vanton4_paired_dpo_s1vs2). Re-administers the v5 + trait_ocean_natural_v1
# questionnaires on a subsample of the B rollout with the LoRA loaded, and
# computes paired F0–F3 deltas vs. the cached baseline scores.
#
# Output → scratch/factor_inspect/validate/control_pdpo_s1vs2/
# Log    → scratch/logs/control_pdpo_s1vs2_validate_<UTC stamp>.log
#
# Single GPU. Expect ~20–60 min depending on GPU type (paired with the warmth
# amp/sup runs which also used 200 personas × 2 questionnaires).
#
# Usage:
#   bash scripts_dev/oct_pipeline/unsup_4fac/run_validate_control_pdpo.sh <gpu_id>
#
# Run via tmux when ready:
#   tmux new-session -d -s ctrl_pdpo_eval \
#     'cd /root/persona-shattering-lasr && bash scripts_dev/oct_pipeline/unsup_4fac/run_validate_control_pdpo.sh 0; exec bash'
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

GPU="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
# Force unbuffered stdout/stderr so tee streams in real time.
export PYTHONUNBUFFERED=1

ADAPTER="persona-shattering-lasr/monorepo::fine_tuning/llama-3.1-8b-it/other/ocean_def_control/amplifier/vanton4_paired_dpo_s1vs2/lora/ocean_def_control_full_vanton4-persona"
LABEL="control_pdpo_s1vs2"
N_PERSONAS=200

LOG_DIR="scratch/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_LOG="${LOG_DIR}/${LABEL}_validate_${STAMP}.log"

echo "================================================================"
echo "  validate_warmth_lora — ${LABEL}"
echo "  GPU:        ${GPU}"
echo "  adapter:    ${ADAPTER}"
echo "  n_personas: ${N_PERSONAS}"
echo "  log:        ${RUN_LOG}"
echo "================================================================"

stdbuf -oL -eL uv run python scripts_dev/oct_pipeline/unsup_4fac/validate_warmth_lora.py \
    --adapter "$ADAPTER" \
    --n-personas "$N_PERSONAS" \
    --label "$LABEL" \
    2>&1 | stdbuf -oL -eL tee "$RUN_LOG"

echo
echo "================================================================"
echo "  ✓ ${LABEL} validation complete"
echo "  Summary:    scratch/factor_inspect/validate/${LABEL}/${LABEL}_summary.json"
echo "  Scores npz: scratch/factor_inspect/validate/${LABEL}/${LABEL}_scores.npz"
echo "  Violin png: scratch/factor_inspect/validate/${LABEL}/${LABEL}_paired_diff.png"
echo "================================================================"
