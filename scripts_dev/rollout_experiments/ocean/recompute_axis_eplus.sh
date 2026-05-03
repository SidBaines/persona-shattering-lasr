#!/usr/bin/env bash
# Recompute the e_plus activation axis against the current canonical
# vanton4_paired_dpo LoRA, replacing the methodologically-iffy older
# vanton1-derived axis.
#
# After this finishes you'll need to manually update OCEAN_REGISTRY (or
# the axis_hf_uri property) in src_dev/common/lora_catalogue.py to point
# at the new axis location. The new artifacts land at:
#
#   persona-shattering-lasr/monorepo/
#     fine_tuning/llama-3.1-8b-it/ocean/extraversion/amplifier/vanton4_paired_dpo/
#       activation_capping/
#         e_plus_axis.pt
#         e_plus_per_layer_range.pt
#         e_plus_activations.pt
#         run_info.json + plots
#
# Estimated time on A100 80GB: ~30-45 minutes (5 rollouts × ~150 prompts ×
# 2 conditions = ~1500 generations, plus 2 forward passes per generation
# for activation extraction).
#
# Usage:
#   bash scripts_dev/rollout_experiments/ocean/recompute_axis_eplus.sh
#
# After it succeeds:
#   1. Update src_dev/common/lora_catalogue.py: change OceanTraitDef's
#      axis_hf_uri / per_layer_range_hf_uri to use the co-located path
#      (under the LoRA's parent dir) instead of the global one.
#   2. Commit + push.
#   3. Run actcap_runs.sh.

set -uo pipefail

cd "$(dirname "$0")/../../.."

git pull --ff-only || echo "WARN: git pull failed — continuing with current checkout"

mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/recompute_axis_eplus_${TS}.log"

echo "[$(date)] === recompute e_plus axis (vanton4_paired_dpo) ===" | tee -a "$LOG"
echo "[$(date)] log: $LOG" | tee -a "$LOG"

# Use --force in case there's already a stale artifact at that path.
uv run python scripts_dev/activation_capping/ocean/paper_versions/compute_axis.py \
    --persona e_plus \
    --force \
    2>&1 | tee -a "$LOG"

EXIT=${PIPESTATUS[0]}
if [ "$EXIT" -eq 0 ]; then
    echo "" | tee -a "$LOG"
    echo "[$(date)] === e_plus axis recomputed ===" | tee -a "$LOG"
    echo "" | tee -a "$LOG"
    echo "  NEXT STEP: update src_dev/common/lora_catalogue.py:" | tee -a "$LOG"
    echo "    Change the axis_hf_uri / per_layer_range_hf_uri properties" | tee -a "$LOG"
    echo "    on OceanTraitDef to point at the co-located path:" | tee -a "$LOG"
    echo "" | tee -a "$LOG"
    echo "      hf://persona-shattering-lasr/monorepo/{lora_parent_dir}/activation_capping/{slug}_axis.pt" | tee -a "$LOG"
    echo "" | tee -a "$LOG"
    echo "    where {lora_parent_dir} = adapter_path_in_repo without the trailing /lora/<adapter>." | tee -a "$LOG"
    echo "    For e_plus that's:" | tee -a "$LOG"
    echo "      hf://persona-shattering-lasr/monorepo/fine_tuning/llama-3.1-8b-it/ocean/extraversion/amplifier/vanton4_paired_dpo/activation_capping/e_plus_axis.pt" | tee -a "$LOG"
    echo "" | tee -a "$LOG"
    echo "    Then commit + push, RunPod git pulls, and run actcap_runs.sh." | tee -a "$LOG"
else
    echo "[$(date)] === FAILED with exit=$EXIT ===" | tee -a "$LOG"
    exit "$EXIT"
fi
