#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# OCEAN vanton4_paired_dpo — "train high, test low" across multiple ranks.
#
# Takes the already-trained vanton4_paired_dpo souped `-persona` adapters,
# applies truncated-SVD rank reduction at each rank in RANKS, and runs the
# trait + MMLU sweeps on the reduced adapters. No retraining — eval-time only.
#
# Edit RANKS and PERSONAS below to choose what runs.
#
# Eval outputs land alongside the existing vanton4_paired_dpo evals on HF,
# with a ``_downrank{N}`` suffix on the eval name (variants of vanton4_paired_dpo,
# not a new monorepo version):
#   fine_tuning/.../<trait>/<direction>/vanton4_paired_dpo/evals/mcq/
#     trait_logprobs_downrank{N}/
#     mmlu_downrank{N}/
#
# Usage:
#   bash scripts_dev/oct_pipeline/ocean/vanton4_paired_dpo_downranks/run_all.sh
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

# Edit these two arrays to control the sweep.
RANKS=(1 2 4 8)
PERSONAS=(o_plus o_minus c_plus c_minus e_plus e_minus a_plus a_minus n_plus n_minus)

DRIVER="scripts_dev/oct_pipeline/ocean/vanton4_paired_dpo_downranks/reduce_and_eval.py"

DONE=()
FAILED=()

for rank in "${RANKS[@]}"; do
    for persona in "${PERSONAS[@]}"; do
        echo ""
        echo "================================================================"
        echo "  rank=${rank}  persona=${persona}"
        echo "================================================================"
        if uv run python "$DRIVER" --rank "$rank" --persona "$persona"; then
            DONE+=("rank=${rank}/${persona}")
        else
            echo "  WARNING: rank=${rank} persona=${persona} failed — continuing"
            FAILED+=("rank=${rank}/${persona}")
        fi
    done
done

echo ""
echo "================================================================"
echo "  Done   (${#DONE[@]}): ${DONE[*]:-none}"
echo "  Failed (${#FAILED[@]}): ${FAILED[*]:-none}"
echo "================================================================"

[ "${#FAILED[@]}" -eq 0 ]
