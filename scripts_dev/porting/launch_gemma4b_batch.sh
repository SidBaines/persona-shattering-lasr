#!/bin/bash
# Launch parallel gemma-3-4b-it LoRA training+eval jobs, one per GPU.
#
# Supports two run types in a single batch:
#   - Ported runs (source_model set): copy teacher distillation data from an
#     existing source-model monorepo path, then run student distillation only.
#   - New-constitution runs (source_model = "-"): run full distillation
#     (teacher + student) from scratch via OpenRouter.
#
# Prerequisites:
#   - Run setup_machine.sh first (or pass --setup to run it automatically)
#   - .env with HF_TOKEN set (and OPENROUTER_API_KEY if any runs have source_model="-")
#
# Usage:
#   bash scripts_dev/porting/launch_gemma4b_batch.sh [--setup]

set -euo pipefail

# Unbuffered Python stdout/stderr so per-job logs flush live when redirected
# to a file. Without this, Python's stderr is block-buffered and log lines can
# appear minutes late, making live progress monitoring misleading.
export PYTHONUNBUFFERED=1

TARGET_MODEL="gemma-3-4b-it"
TEACHER="z-ai/glm-4.5-air"
MAX_LEN=2048
SAMPLES_PER_TRAIT=300
MMLU_LIMIT=500
LOG_DIR="scratch/logs/gemma4b_batch"

# =====================================================================
# Optional: run setup first
# =====================================================================
if [[ "${1:-}" == "--setup" ]]; then
    echo "======================================================================"
    echo "  Running machine setup..."
    echo "======================================================================"
    bash scripts_dev/porting/setup_machine.sh
    echo ""
fi

mkdir -p "$LOG_DIR"

# =====================================================================
# Define the runs
# =====================================================================
# Format: "trait  direction  version  constitution_name  introspection_constitution  source_model"
# introspection_constitution: slim variant for introspection/SFT stages (use "-" for same as main)
# source_model: monorepo path to port teacher data from (use "-" to skip copy and run full distillation from scratch)
RUNS=(
    "neuroticism    suppressor  4   neuroticism_low                        -                                              llama-3.1-8b-it"
    "extraversion   suppressor  anton1  extraversion_suppressing_full_vanton1  extraversion_suppressing_full_vanton1_slim  llama-3.1-8b-it"
    "agreeableness  amplifier   s1  agreeableness_high_s1                  -                                              -"
    "agreeableness  suppressor  s1  agreeableness_low_s1                   -                                              -"
)

# Previous run-set (kept for reference):
# RUNS=(
#     "openness       amplifier   anton1  openness_amplifying_full_vanton1       openness_amplifying_full_vanton1_slim"
#     "neuroticism    amplifier   anton1  neuroticism_amplifying_full_vanton1    neuroticism_amplifying_full_vanton1_slim"
#     "extraversion   amplifier   anton1  extraversion_amplifying_full_vanton1   extraversion_amplifying_full_vanton1_slim"
#     "agreeableness  suppressor  2       agreeableness_low                      -"
#     "openness       suppressor  anton1  openness_suppressing_full_vanton1      openness_suppressing_full_vanton1_slim"
# )

# =====================================================================
# Step 1: Copy teacher data for runs with a source_model (skip "-")
# =====================================================================
echo ""
echo "======================================================================"
echo "  Step 1: Copying teacher data (only for runs with a source_model)"
echo "======================================================================"

for run_spec in "${RUNS[@]}"; do
    read -r trait direction version constitution intro_constitution source_model <<< "$run_spec"

    if [[ "${source_model}" == "-" ]]; then
        echo ""
        echo "  Skipping copy for ${trait}/${direction}/v${version} (${constitution}) — no source model, will run full distillation from scratch"
        continue
    fi

    echo ""
    echo "  Copying: ${trait}/${direction}/v${version} (${constitution}) from ${source_model}"
    uv run python scripts_dev/porting/copy_teacher_data.py \
        --source-model "${source_model}" \
        --target-model "${TARGET_MODEL}" \
        --trait "${trait}" \
        --direction "${direction}" \
        --version "${version}" \
        --constitution "${constitution}"
done

echo ""
echo "======================================================================"
echo "  Step 2: Launching ${#RUNS[@]} parallel training+eval jobs"
echo "======================================================================"

# =====================================================================
# Step 2: Launch parallel jobs (one per GPU)
# =====================================================================
PIDS=()
GPU=0

for run_spec in "${RUNS[@]}"; do
    read -r trait direction version constitution intro_constitution source_model <<< "$run_spec"
    log_file="${LOG_DIR}/${trait}_${direction}_v${version}.log"

    echo ""
    echo "  GPU ${GPU}: ${trait}/${direction}/v${version} -> ${log_file}"

    EXTRA_ARGS=()
    if [[ "${intro_constitution}" != "-" ]]; then
        EXTRA_ARGS+=(--introspection-constitution "scripts_dev/oct_pipeline/ocean/${intro_constitution}.json")
    fi

    CUDA_VISIBLE_DEVICES=${GPU} MASTER_PORT=$((29500 + GPU)) bash scripts_dev/oct_pipeline/run_ocean_persona_e2e.sh \
        --constitution "scripts_dev/oct_pipeline/ocean/${constitution}.json" \
        --trait "${trait}" \
        --direction "${direction}" \
        --version "${version}" \
        --model "${TARGET_MODEL}" \
        --teacher "${TEACHER}" \
        --max-len "${MAX_LEN}" \
        --samples-per-trait "${SAMPLES_PER_TRAIT}" \
        --mmlu-limit "${MMLU_LIMIT}" \
        "${EXTRA_ARGS[@]}" \
        > "${log_file}" 2>&1 &

    PIDS+=($!)
    GPU=$((GPU + 1))
done

echo ""
echo "======================================================================"
echo "  All ${#RUNS[@]} jobs launched. PIDs: ${PIDS[*]}"
echo "  Logs: ${LOG_DIR}/"
echo ""
echo "  Monitor with:"
echo "    tail -f ${LOG_DIR}/*.log"
echo "    nvidia-smi -l 5"
echo "======================================================================"

# =====================================================================
# Wait for all jobs and report results
# =====================================================================
FAILED=0
for i in "${!PIDS[@]}"; do
    read -r trait direction version constitution intro_constitution source_model <<< "${RUNS[$i]}"
    pid="${PIDS[$i]}"
    log_file="${LOG_DIR}/${trait}_${direction}_v${version}.log"

    if wait "$pid"; then
        echo "  [OK]   GPU $i: ${trait}/${direction}/v${version}"
    else
        echo "  [FAIL] GPU $i: ${trait}/${direction}/v${version} (see ${log_file})"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
if [[ $FAILED -eq 0 ]]; then
    echo "======================================================================"
    echo "  All ${#RUNS[@]} jobs completed successfully."
    echo "======================================================================"
else
    echo "======================================================================"
    echo "  ${FAILED}/${#RUNS[@]} jobs failed. Check logs in ${LOG_DIR}/"
    echo "======================================================================"
    exit 1
fi
