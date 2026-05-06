#!/usr/bin/env bash
# Run all 4 remaining same-trait opposing LoRA sweeps sequentially, then
# plot each heatmap as soon as its sweep completes and data is on HF.
#
# Openness is excluded — run it separately first:
#   CUDA_VISIBLE_DEVICES=0 uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \
#       --config scripts_dev.evals.llm_judge_sweep.configs.vanton4_paired_dpo.o_plus_x_o_minus_on_openness \
#       --allow-custom-fingerprint 2>&1 | tee scratch/sweep_o_plus_x_o_minus.log
#
# Usage (from repo root):
#   CUDA_VISIBLE_DEVICES=0 bash scripts_dev/evals/llm_judge_sweep/run_opposing_lora_sweeps.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="scratch"
mkdir -p "$LOG_DIR"

# Extract the 10-char fingerprint from a completed sweep log.
# The runner prints: sweep_id=<eval_name>_<fingerprint>_<uuid8>
# The fingerprint is the 10-char hex segment immediately before the 8-char uuid suffix.
extract_fingerprint() {
    local log_file="$1"
    grep -m1 "sweep_id=" "$log_file" \
        | grep -oP '(?<=sweep_id=)[^\s]+' \
        | grep -oP '[0-9a-f]{10}(?=_[0-9a-f]{8}$)'
}

run_sweep_and_plot() {
    local trait="$1"          # e.g. "conscientiousness"
    local config_module="$2"  # e.g. "c_plus_x_c_minus_on_conscientiousness"
    local log_file="$LOG_DIR/sweep_${config_module}.log"

    echo ""
    echo "========================================================"
    echo "  Starting sweep: ${config_module}"
    echo "  Log: ${log_file}"
    echo "========================================================"

    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \
        --config "scripts_dev.evals.llm_judge_sweep.configs.vanton4_paired_dpo.${config_module}" \
        --allow-custom-fingerprint \
        2>&1 | tee "$log_file"

    # Check the runner exited cleanly (tee always exits 0, so inspect log).
    if ! grep -q "^Done\." "$log_file"; then
        echo "ERROR: sweep ${config_module} did not complete cleanly. Check ${log_file}."
        exit 1
    fi

    local fp
    fp="$(extract_fingerprint "$log_file")"
    if [[ -z "$fp" ]]; then
        echo "ERROR: could not extract fingerprint from ${log_file}."
        exit 1
    fi

    echo ""
    echo "  Sweep complete. fingerprint=${fp}"
    echo "  Rendering heatmap for ${trait} ..."

    uv run python -m src_dev.visualisations.paper_appendix_opposing_lora_heatmaps \
        "--${trait}" "$fp" \
        2>&1 | tee "${LOG_DIR}/plot_opposing_${trait}.log"

    echo "  Heatmap done for ${trait}."
}

run_sweep_and_plot "conscientiousness" "c_plus_x_c_minus_on_conscientiousness"
run_sweep_and_plot "extraversion"      "e_plus_x_e_minus_on_extraversion"
run_sweep_and_plot "agreeableness"     "a_plus_x_a_minus_on_agreeableness"
run_sweep_and_plot "neuroticism"       "n_plus_x_n_minus_on_neuroticism"

echo ""
echo "All 4 sweeps complete. Figures written to paper/figures/appendix/."
