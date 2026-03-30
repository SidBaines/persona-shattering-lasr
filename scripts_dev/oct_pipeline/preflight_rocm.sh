#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_PATH="${OCT_VENV_PATH:-${REPO_ROOT}/.venv-oct}"
SKIP_TORCH_PROBE="${OCT_PREFLIGHT_SKIP_TORCH_PROBE:-0}"
IGNORE_DSTATE="${OCT_PREFLIGHT_IGNORE_DSTATE:-0}"

if [[ "${1:-}" == "--skip-torch-probe" ]]; then
  SKIP_TORCH_PROBE=1
fi
if [[ "${1:-}" == "--ignore-dstate" ]] || [[ "${2:-}" == "--ignore-dstate" ]]; then
  IGNORE_DSTATE=1
fi

if [[ ! -d "${VENV_PATH}" ]]; then
  echo "[FAIL] Missing venv at ${VENV_PATH}" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${VENV_PATH}/bin/activate"

echo "== ROCm Preflight =="
echo "repo_root=${REPO_ROOT}"
echo "venv=${VENV_PATH}"
echo "python=$(command -v python)"

run_with_timeout() {
  local timeout_seconds="$1"
  shift
  python - "$timeout_seconds" "$@" <<'PY'
import subprocess
import sys
import time

timeout_seconds = float(sys.argv[1])
cmd = sys.argv[2:]
proc = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
)

deadline = time.monotonic() + timeout_seconds
while time.monotonic() < deadline:
    rc = proc.poll()
    if rc is not None:
        out, err = proc.communicate()
        if out:
            sys.stdout.write(out)
        if err:
            sys.stderr.write(err)
        sys.exit(rc)
    time.sleep(0.2)

print(
    f"[FAIL] Timeout after {timeout_seconds:.0f}s: {' '.join(cmd)}",
    file=sys.stderr,
)
try:
    proc.kill()
except Exception:
    pass
sys.exit(124)
PY
}

# Fail fast if previous uninterruptible GPU hangs are already present.
stuck_processes="$(ps -eo pid,stat,cmd | awk '
  $2 ~ /D/ && ($3 ~ /rocminfo|python/ || $0 ~ /torch|hip|rocm/) {print}
')"
if [[ -n "${stuck_processes}" ]]; then
  if [[ "${IGNORE_DSTATE}" == "1" ]]; then
    echo "[WARN] Ignoring existing D-state GPU-related processes due override:"
    echo "${stuck_processes}"
  else
    echo "[FAIL] Found D-state GPU-related processes. Runtime already wedged:" >&2
    echo "${stuck_processes}" >&2
    exit 1
  fi
fi

if pgrep -f "src_dev/utils/gpustat.py" >/dev/null 2>&1; then
  echo "[WARN] gpustat monitor is running. This is usually fine, but disable if debugging instability."
fi

echo "[1/2] rocm-smi quick probe..."
run_with_timeout 15 rocm-smi --showuse --showmemuse --showtemp --csv >/tmp/oct_rocm_smi_preflight.txt
tail -n +1 /tmp/oct_rocm_smi_preflight.txt | head -n 20

if [[ "${SKIP_TORCH_PROBE}" == "1" ]]; then
  echo "[2/2] torch ROCm probe skipped (OCT_PREFLIGHT_SKIP_TORCH_PROBE=1)"
else
  echo "[2/2] torch ROCm probe (isolated child process)..."
  run_with_timeout 20 python -c "import torch; print('torch:', torch.__version__); print('hip:', torch.version.hip); print('cuda_available:', torch.cuda.is_available()); print('device_count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"
fi

echo "== Preflight PASS =="
