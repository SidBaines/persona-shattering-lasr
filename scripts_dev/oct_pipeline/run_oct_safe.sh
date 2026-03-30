#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_PATH="${OCT_VENV_PATH:-${REPO_ROOT}/.venv-oct}"
SKIP_TORCH_PREFLIGHT=0
IGNORE_DSTATE=0

while [[ "${1:-}" == --* ]]; do
  case "$1" in
    --skip-torch-preflight)
      SKIP_TORCH_PREFLIGHT=1
      shift
      ;;
    --ignore-dstate)
      IGNORE_DSTATE=1
      shift
      ;;
    *)
      break
      ;;
  esac
done

if [[ "${1:-}" == "" ]]; then
  cat <<'USAGE'
Usage:
  scripts_dev/oct_pipeline/run_oct_safe.sh [--skip-torch-preflight] [--ignore-dstate] <run_oct_pipeline.py args...>

Example:
  scripts_dev/oct_pipeline/run_oct_safe.sh \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --model-path /root/.cache/models \
    --teacher-model z-ai/glm-4.5-air \
    --constitution conscientiousness_low \
    --custom-constitution scripts_dev/oct_pipeline/conscientiousness_low_v3.json \
    --training-backend oct \
    --vllm-gpu-memory-utilization 0.35 \
    --oct-dpo-micro-batch-size 1 \
    --oct-sft-micro-batch-size 1 \
    --seed 31003 \
    --out-dir scratch/oct_parallel_llama31_8b
USAGE
  exit 1
fi

if [[ ! -d "${VENV_PATH}" ]]; then
  echo "[FAIL] Missing venv at ${VENV_PATH}" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${VENV_PATH}/bin/activate"

echo "== OCT Safe Launcher =="
echo "repo_root=${REPO_ROOT}"
echo "venv=${VENV_PATH}"
echo "python=$(command -v python)"
echo

python "${SCRIPT_DIR}/check_oct_env.py" \
  --repo-root "${REPO_ROOT}" \
  --model-path /root/.cache/models

if [[ "${SKIP_TORCH_PREFLIGHT}" == "1" ]]; then
  if [[ "${IGNORE_DSTATE}" == "1" ]]; then
    OCT_PREFLIGHT_SKIP_TORCH_PROBE=1 OCT_PREFLIGHT_IGNORE_DSTATE=1 \
      "${SCRIPT_DIR}/preflight_rocm.sh" --skip-torch-probe --ignore-dstate
  else
    OCT_PREFLIGHT_SKIP_TORCH_PROBE=1 \
      "${SCRIPT_DIR}/preflight_rocm.sh" --skip-torch-probe
  fi
else
  if [[ "${IGNORE_DSTATE}" == "1" ]]; then
    OCT_PREFLIGHT_IGNORE_DSTATE=1 "${SCRIPT_DIR}/preflight_rocm.sh" --ignore-dstate
  else
    "${SCRIPT_DIR}/preflight_rocm.sh"
  fi
fi

echo
echo "Launching pipeline with args:"
printf '  %q' "$@"
echo
echo

exec python "${SCRIPT_DIR}/run_oct_pipeline.py" "$@"
