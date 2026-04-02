#!/bin/bash
# End-to-end OCEAN persona pipeline: distillation → training → evals.
#
# Usage:
#   # Full pipeline (training + TRAIT eval + MMLU eval)
#   bash scripts_dev/oct_pipeline/run_ocean_persona_e2e.sh \
#       --constitution scripts_dev/oct_pipeline/ocean/agreeableness_low.json \
#       --trait agreeableness --direction suppressor --version 3 \
#       --teacher meta-llama/llama-3.1-8b-instruct
#
#   # Distillation only
#   bash scripts_dev/oct_pipeline/run_ocean_persona_e2e.sh \
#       --constitution scripts_dev/oct_pipeline/ocean/agreeableness_low.json \
#       --trait agreeableness --direction suppressor --version 3 \
#       --teacher meta-llama/llama-3.1-8b-instruct \
#       --stop-after distillation
#
#   # Evals only (adapter already on HF)
#   bash scripts_dev/oct_pipeline/run_ocean_persona_e2e.sh \
#       --trait agreeableness --direction suppressor --version 2 \
#       --constitution scripts_dev/oct_pipeline/ocean/agreeableness_low.json \
#       --skip-to evals
#
#   # Training + TRAIT only (skip MMLU)
#   bash scripts_dev/oct_pipeline/run_ocean_persona_e2e.sh \
#       --constitution scripts_dev/oct_pipeline/ocean/agreeableness_low.json \
#       --trait agreeableness --direction suppressor --version 3 \
#       --teacher meta-llama/llama-3.1-8b-instruct \
#       --skip-mmlu

set -euo pipefail

# =====================================================================
# Defaults
# =====================================================================
MODEL=llama-3.1-8b-it
MODEL_PATH=/root/.cache/models
TEACHER=z-ai/glm-4.5-air
LORA_RANK=64
LORA_ALPHA=128
LEARNING_RATE=5e-5
BETA=0.1
SEED=123456
SAMPLES_PER_TRAIT=300
MMLU_LIMIT=300
BATCH_SIZE=128
STOP_AFTER=""
SKIP_TO=""
SKIP_TRAIT=false
SKIP_MMLU=false

# Required (no defaults)
CONSTITUTION=""
TRAIT=""
DIRECTION=""
VERSION=""

# =====================================================================
# Parse args
# =====================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --constitution) CONSTITUTION="$2"; shift 2 ;;
        --trait) TRAIT="$2"; shift 2 ;;
        --direction) DIRECTION="$2"; shift 2 ;;
        --version) VERSION="$2"; shift 2 ;;
        --teacher) TEACHER="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --lora-rank) LORA_RANK="$2"; shift 2 ;;
        --lora-alpha) LORA_ALPHA="$2"; shift 2 ;;
        --learning-rate) LEARNING_RATE="$2"; shift 2 ;;
        --beta) BETA="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --samples-per-trait) SAMPLES_PER_TRAIT="$2"; shift 2 ;;
        --mmlu-limit) MMLU_LIMIT="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --stop-after) STOP_AFTER="$2"; shift 2 ;;
        --skip-to) SKIP_TO="$2"; shift 2 ;;
        --skip-trait) SKIP_TRAIT=true; shift ;;
        --skip-mmlu) SKIP_MMLU=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Validate required args
for var in CONSTITUTION TRAIT DIRECTION VERSION; do
    if [[ -z "${!var}" ]]; then
        echo "Error: --$(echo $var | tr '[:upper:]' '[:lower:]') is required"
        exit 1
    fi
done

# Derived values
CONSTITUTION_NAME=$(basename "${CONSTITUTION}" .json)
TRAIT_ABBREV=$(echo "${TRAIT}" | cut -c1 | tr '[:lower:]' '[:upper:]')
if [[ "${DIRECTION}" == "amplifier" ]]; then
    DIRECTION_TAG="+"
else
    DIRECTION_TAG="-"
fi
RUN_NAME="${TRAIT_ABBREV}${DIRECTION_TAG}_v${VERSION}"
HF_REPO="persona-shattering-lasr/monorepo"
ADAPTER_HF_PATH="fine_tuning/${MODEL}/ocean/${TRAIT}/${DIRECTION}/v${VERSION}/lora/${CONSTITUTION_NAME}-persona"

echo ""
echo "======================================================================"
echo "  OCEAN Persona E2E: ${RUN_NAME}"
echo "  Constitution: ${CONSTITUTION}"
echo "  Teacher: ${TEACHER}"
echo "  Adapter: ${HF_REPO}/${ADAPTER_HF_PATH}"
echo "======================================================================"

# =====================================================================
# Stage 1: OCT Pipeline (distillation + training)
# =====================================================================
if [[ "${SKIP_TO}" != "evals" ]]; then
    echo ""
    echo "======================================================================"
    echo "  Stage 1: OCT Pipeline"
    echo "======================================================================"
    echo ""

    OCT_ARGS=(
        --model "${MODEL}"
        --model-path "${MODEL_PATH}"
        --teacher-model "${TEACHER}"
        --custom-constitution "${CONSTITUTION}"
        --lora-rank "${LORA_RANK}"
        --lora-alpha "${LORA_ALPHA}"
        --learning-rate "${LEARNING_RATE}"
        --beta "${BETA}"
        --seed "${SEED}"
        --monorepo-category ocean
        --monorepo-trait "${TRAIT}"
        --monorepo-direction "${DIRECTION}"
        --monorepo-version "${VERSION}"
    )

    if [[ "${STOP_AFTER}" == "distillation" ]]; then
        OCT_ARGS+=(--stages distillation --skip-training)
    fi

    uv run --with-requirements scripts_dev/oct_pipeline/uv-oct-requirements.txt \
        python scripts_dev/oct_pipeline/run_oct_pipeline.py "${OCT_ARGS[@]}"

    if [[ -n "${STOP_AFTER}" ]]; then
        echo ""
        echo "  Stopped after ${STOP_AFTER}."
        exit 0
    fi

    # Cleanup distilled model (~16GB)
    echo ""
    echo "  Cleaning up distilled model..."
    rm -rf scratch/oct_runs/*/models/distilled/ 2>/dev/null || true
    rm -rf "${MODEL_PATH}/${MODEL}-"*"${CONSTITUTION_NAME}"* 2>/dev/null || true
    echo "  Done."
fi

# =====================================================================
# Generate eval config
# =====================================================================
# Write a temporary eval config with the correct adapter path.
# This avoids maintaining separate config files per version.
EVAL_CONFIG_DIR="scratch/eval_configs"
mkdir -p "${EVAL_CONFIG_DIR}"

_write_eval_config() {
    local eval_type="$1"  # trait or mmlu
    local config_file="${EVAL_CONFIG_DIR}/${RUN_NAME}_${eval_type}.py"

    if [[ "${eval_type}" == "trait" ]]; then
        cat > "${config_file}" << PYEOF
"""Auto-generated TRAIT eval config for ${RUN_NAME}."""
from pathlib import Path
from dotenv import load_dotenv
from src_dev.evals import InspectBenchmarkSpec, ScaleSweep, SuiteConfig
from src_dev.utils.hf_hub import download_from_dataset_repo
load_dotenv()

_HF_REPO = "${HF_REPO}"
_PATH_IN_REPO = "${ADAPTER_HF_PATH}"
_CACHE = Path("scratch/adapters/${RUN_NAME}")
download_from_dataset_repo(repo_id=_HF_REPO, path_in_repo=_PATH_IN_REPO, local_dir=_CACHE)
_ADAPTER_URI = f"local://{(_CACHE / _PATH_IN_REPO).resolve()}"

def _scale_points():
    c_neg = [round(-4.0 + i * 0.5, 10) for i in range(round((-2.5 - -4.0) / 0.5) + 1)]
    fine = [round(-2.0 + i * 0.25, 10) for i in range(round((2.0 - -2.0) / 0.25) + 1)]
    c_pos = [round(2.5 + i * 0.5, 10) for i in range(round((4.0 - 2.5) / 0.5) + 1)]
    return sorted({s for s in c_neg + fine + c_pos if s != 0.0})

SUITE_CONFIG = SuiteConfig(
    base_model="meta-llama/Llama-3.1-8B-Instruct",
    adapter=_ADAPTER_URI,
    sweep=ScaleSweep(points=_scale_points()),
    evals=[InspectBenchmarkSpec(
        name="trait", benchmark="personality_trait_sampled",
        benchmark_args={"samples_per_trait": ${SAMPLES_PER_TRAIT},
                        "trait_splits": ["Openness","Conscientiousness","Extraversion","Agreeableness","Neuroticism"],
                        "max_tokens": 32},
        n_runs=1,
    )],
    temperature=0.0, batch_size=${BATCH_SIZE},
    output_root=Path("scratch/evals/ocean/trait"),
    run_name="${RUN_NAME}",
    skip_completed=True, auto_analyze=True,
    analyze_kwargs={"title_suffix": "${RUN_NAME} TRAIT", "interval": "ci95_from_wilson"},
    upload_repo_id=_HF_REPO,
    upload_path_in_repo="fine_tuning/${MODEL}/ocean/${TRAIT}/evals/mcq/trait/${RUN_NAME}",
    metadata={"persona": "${RUN_NAME}", "adapter": f"{_HF_REPO}::{_PATH_IN_REPO}"},
)
PYEOF
    else
        cat > "${config_file}" << PYEOF
"""Auto-generated MMLU eval config for ${RUN_NAME}."""
from pathlib import Path
from dotenv import load_dotenv
from src_dev.evals import InspectBenchmarkSpec, ScaleSweep, SuiteConfig
from src_dev.utils.hf_hub import download_from_dataset_repo
load_dotenv()

_HF_REPO = "${HF_REPO}"
_PATH_IN_REPO = "${ADAPTER_HF_PATH}"
_CACHE = Path("scratch/adapters/${RUN_NAME}")
download_from_dataset_repo(repo_id=_HF_REPO, path_in_repo=_PATH_IN_REPO, local_dir=_CACHE)
_ADAPTER_URI = f"local://{(_CACHE / _PATH_IN_REPO).resolve()}"

_SCALE = sorted({round(-4.0 + i * 0.5, 10) for i in range(17) if round(-4.0 + i * 0.5, 10) != 0.0})

SUITE_CONFIG = SuiteConfig(
    base_model="meta-llama/Llama-3.1-8B-Instruct",
    adapter=_ADAPTER_URI,
    sweep=ScaleSweep(points=_SCALE),
    evals=[InspectBenchmarkSpec(
        name="mmlu", benchmark="mmlu", limit=${MMLU_LIMIT}, n_runs=1,
    )],
    temperature=0.0, batch_size=${BATCH_SIZE},
    output_root=Path("scratch/evals/ocean/mmlu"),
    run_name="${RUN_NAME}_mmlu",
    skip_completed=True, auto_analyze=True,
    analyze_kwargs={"random_baseline": 0.25, "title_suffix": "${RUN_NAME} MMLU", "interval": "ci95_from_wilson"},
    upload_repo_id=_HF_REPO,
    upload_path_in_repo="fine_tuning/${MODEL}/ocean/${TRAIT}/evals/mcq/mmlu/${RUN_NAME}",
    metadata={"persona": "${RUN_NAME}", "adapter": f"{_HF_REPO}::{_PATH_IN_REPO}"},
)
PYEOF
    fi
    echo "${config_file}"
}

# =====================================================================
# Stage 2: TRAIT sweep
# =====================================================================
if [[ "${SKIP_TRAIT}" != "true" ]]; then
    TRAIT_CONFIG=$(_write_eval_config trait)
    # Convert file path to Python module path
    TRAIT_MODULE=$(echo "${TRAIT_CONFIG}" | sed 's|/|.|g; s|\.py$||')

    echo ""
    echo "======================================================================"
    echo "  Stage 2: TRAIT sweep — ${RUN_NAME}"
    echo "======================================================================"
    echo ""

    uv run python -m src_dev.evals suite --config-module "${TRAIT_MODULE}"
fi

# =====================================================================
# Stage 3: MMLU sweep
# =====================================================================
if [[ "${SKIP_MMLU}" != "true" ]]; then
    MMLU_CONFIG=$(_write_eval_config mmlu)
    MMLU_MODULE=$(echo "${MMLU_CONFIG}" | sed 's|/|.|g; s|\.py$||')

    echo ""
    echo "======================================================================"
    echo "  Stage 3: MMLU sweep — ${RUN_NAME}"
    echo "======================================================================"
    echo ""

    uv run python -m src_dev.evals suite --config-module "${MMLU_MODULE}"
fi

echo ""
echo "======================================================================"
echo "  All stages complete: ${RUN_NAME}"
echo "======================================================================"
