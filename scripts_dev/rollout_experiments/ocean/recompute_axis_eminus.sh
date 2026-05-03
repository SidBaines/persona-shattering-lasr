#!/usr/bin/env bash
# Recompute the e_minus activation axis against the canonical
# vanton4_paired_dpo LoRA. We have an e_plus axis but no e_minus axis;
# the overnight steering matrix needs E- direction actcap, so this is
# the prerequisite.
#
# Artifacts will land at:
#
#   persona-shattering-lasr/monorepo/
#     fine_tuning/llama-3.1-8b-it/ocean/extraversion/suppressor/vanton4_paired_dpo/
#       activation_capping/
#         e_minus_axis.pt
#         e_minus_per_layer_range.pt
#         e_minus_activations.pt
#         run_info.json + plots
#
# OCEAN_REGISTRY's `e_minus` entry has axis_slug=None today; once this
# completes the next session needs to set axis_slug="e_minus" so
# axis_hf_uri / per_layer_range_hf_uri resolve correctly.
#
# Estimated time on A100 80GB: ~30-45 minutes.
#
# Usage:
#   tmux new -s axis_recompute
#   bash scripts_dev/rollout_experiments/ocean/recompute_axis_eminus.sh
#
# After it succeeds:
#   1. In src_dev/common/lora_catalogue.py, set axis_slug="e_minus" on
#      the e_minus OceanTraitDef entry.
#   2. Commit + push.
#   3. Run the overnight matrix.

set -uo pipefail

cd "$(dirname "$0")/../../.."

git pull --ff-only || echo "WARN: git pull failed — continuing with current checkout"

mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/recompute_axis_eminus_${TS}.log"

echo "[$(date)] === recompute e_minus axis (vanton4_paired_dpo) ===" | tee -a "$LOG"
echo "[$(date)] log: $LOG" | tee -a "$LOG"

uv run python scripts_dev/activation_capping/ocean/paper_versions/compute_axis.py \
    --persona e_minus \
    --force \
    2>&1 | tee -a "$LOG"

EXIT=${PIPESTATUS[0]}
if [ "$EXIT" -eq 0 ]; then
    echo "" | tee -a "$LOG"
    echo "[$(date)] === e_minus axis recomputed ===" | tee -a "$LOG"
    echo "" | tee -a "$LOG"
    echo "  NEXT STEP: in src_dev/common/lora_catalogue.py, set" | tee -a "$LOG"
    echo "    axis_slug=\"e_minus\" on the e_minus OceanTraitDef." | tee -a "$LOG"
    echo "  Then commit + push and the overnight matrix can use it." | tee -a "$LOG"
else
    echo "[$(date)] === FAILED with exit=$EXIT ===" | tee -a "$LOG"
    exit "$EXIT"
fi
