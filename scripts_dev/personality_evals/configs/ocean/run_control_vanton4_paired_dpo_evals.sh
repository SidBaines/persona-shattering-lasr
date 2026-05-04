#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Run the full eval suite for the llama-3.1-8b-it recipe-matched null-control
# adapter (vanton4_paired_dpo_s1vs2):
#
#   1. trait MCQ logprobs sweep      (scripts_dev.personality_evals.configs.ocean.trait.vanton4_paired_dpo.control_s1vs2_vanton4_paired_dpo)
#   2. MMLU sweep                    (scripts_dev.personality_evals.configs.ocean.mmlu.vanton4_paired_dpo.control_s1vs2_vanton4_paired_dpo)
#   3. LLM-judge sweeps × 5 traits   (scripts_dev.evals.llm_judge_sweep.configs.vanton4_paired_dpo.control_s1vs2_on_{openness,conscientiousness,extraversion,agreeableness,neuroticism})
#
# Mirrors the eval pattern used for the OCEAN ± vanton4_paired_dpo adapters,
# pointed at the control adapter. Results upload under
#   fine_tuning/llama-3.1-8b-it/other/ocean_def_control/amplifier/vanton4_paired_dpo_s1vs2/evals/
# on persona-shattering-lasr/monorepo.
#
# Each step continues to the next on failure (soft-fail). Logs are tee'd to
# scratch/runner_logs/.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts_dev/personality_evals/configs/ocean/run_control_vanton4_paired_dpo_evals.sh
# ─────────────────────────────────────────────────────────────────────────────
set -o pipefail

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "ERROR: set CUDA_VISIBLE_DEVICES before invoking (LLM-judge runner needs it)."
    exit 1
fi

FAILED_STEPS=()

RUNNER_LOG_DIR="scratch/runner_logs"
mkdir -p "$RUNNER_LOG_DIR"

run_step() {
    local label="$1"; shift
    local safe_label="${label// /_}"
    local log="${RUNNER_LOG_DIR}/${safe_label}.log"
    echo ""
    echo "=== Running: ${label}  (log: ${log}) ==="
    if ! "$@" 2>&1 | tee "$log"; then
        echo "!!! FAILED: ${label} — continuing to next  (log: ${log}) ==="
        FAILED_STEPS+=("$label")
    fi
    echo "=== Done: ${label} ==="
}

# ─── Block 1: MCQ (trait + MMLU) ─────────────────────────────────────────────
run_step "control_s1vs2_trait_logprobs" \
    uv run python -m src_dev.evals suite \
        --config-module scripts_dev.personality_evals.configs.ocean.trait.vanton4_paired_dpo.control_s1vs2_vanton4_paired_dpo

run_step "control_s1vs2_mmlu" \
    uv run python -m src_dev.evals suite \
        --config-module scripts_dev.personality_evals.configs.ocean.mmlu.vanton4_paired_dpo.control_s1vs2_vanton4_paired_dpo

# ─── Block 2: LLM-judge × 5 traits (delegates to existing shard runner) ──────
# The existing runner takes config-name suffixes as positional args, formats
# them as scripts_dev.evals.llm_judge_sweep.configs.vanton4_paired_dpo.<name>,
# and handles batched HF uploads + per-cell sharding internally.
run_step "control_s1vs2_llm_judge_5traits" \
    bash scripts_dev/evals/llm_judge_sweep/run_vanton4_paired_dpo.sh \
        control_s1vs2_on_openness \
        control_s1vs2_on_conscientiousness \
        control_s1vs2_on_extraversion \
        control_s1vs2_on_agreeableness \
        control_s1vs2_on_neuroticism

# ─── Summary ─────────────────────────────────────────────────────────────────
echo ""
if [ ${#FAILED_STEPS[@]} -eq 0 ]; then
    echo "All control_s1vs2 eval steps complete."
    exit 0
else
    echo "${#FAILED_STEPS[@]} step(s) failed:"
    for step in "${FAILED_STEPS[@]}"; do
        echo "  - $step"
    done
    exit 1
fi
