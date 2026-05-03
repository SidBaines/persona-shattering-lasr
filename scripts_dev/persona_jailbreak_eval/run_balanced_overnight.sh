#!/usr/bin/env bash
# Orchestrates the full balanced-scale jailbreak eval suite end-to-end.
#
# Stages (each idempotent — re-runs pick up cached artefacts):
#   1. Drift Phase 1 — build the BASE axis at balanced scale (--upload-hf
#      pushes it to the monorepo so future machines hydrate for free).
#      This produces a much richer axis than smoke (100 roles × 80 q × 3
#      sysprompts) and the auto-picked layer window should now land in the
#      paper's 70-90% depth range, not smoke's noisy 50%.
#   2. Drift Phase 2 — derive capping_config from the balanced axis (CPU,
#      ~30s). Writes the canonical capping_config.pt under the drift
#      scratch dir.
#   3. Persona × StrongREJECT grid eval at balanced (3 conditions × 20
#      personas × 3 sysprompts × 150 harm questions + 200 benign control).
#   4. WildJailbreak eval at balanced (3 conditions × 800 adv-harmful +
#      210 adv-benign, no system prompt).
#
# Both eval scripts auto-upload to HF after each stage (inference,
# judging, aggregation), so all results round-trip through the monorepo.
#
# Compute estimate: ~6-9 GPU hours, ~$60-80 in API costs total. Suitable
# for overnight on a single H100/H200.
#
# Recommended invocation (so a dropped SSH doesn't kill the run):
#   tmux new -s jb_balanced \
#     'bash scripts_dev/persona_jailbreak_eval/run_balanced_overnight.sh'
# Detach with Ctrl+b d; reattach with `tmux attach -t jb_balanced`.

set -euo pipefail

# ── Locate repo root and environment ─────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p logs

if [[ ! -x .venv/bin/python ]]; then
  echo "ERROR: .venv/bin/python not found at ${REPO_ROOT}/.venv/bin/python" >&2
  echo "Make sure the venv is linked per drift HANDOVER §4.2:" >&2
  echo "  ln -snf /root/persona-shattering-lasr/.venv .venv" >&2
  exit 1
fi
if [[ ! -f .env ]]; then
  echo "WARNING: .env not found — HF_TOKEN / OPENROUTER_API_KEY may be missing." >&2
fi

# Canonical paths the drift script writes to (run_slug=balanced_v1 in
# scripts_dev/persona_drift_assistant_axis/config.py:BALANCED).
AXIS_PATH="scratch/persona_drift_assistant_axis/llama-3.1-8b-instruct/balanced_v1/axes/base/axis.pt"
CAPPING_PATH="scratch/persona_drift_assistant_axis/llama-3.1-8b-instruct/balanced_v1/capping_config.pt"

echo "============================================================================"
echo "  Balanced-scale jailbreak eval — full overnight pipeline"
echo "  Repo:    ${REPO_ROOT}"
echo "  Logs:    ${REPO_ROOT}/logs/balanced_*.log"
echo "  Axis:    ${AXIS_PATH}"
echo "  Capping: ${CAPPING_PATH}"
echo "  Started: $(date)"
echo "============================================================================"

# ── 1. Drift Phase 1 — balanced axis (uploads to HF) ─────────────────────
echo
echo ">>> [1/4] Drift Phase 1 — build BASE axis at balanced scale (~2 GPU hr, ~\$15 judge)"
echo "    Idempotent: skips already-completed sub-stages of the upstream pipeline."
echo "    Uploads to monorepo so future machines hydrate axis without recomputing."
.venv/bin/python -m scripts_dev.persona_drift_assistant_axis.build_axis \
    --preset balanced --variant base --upload-hf \
    2>&1 | tee logs/balanced_01_drift_build_axis.log

if [[ ! -f "${AXIS_PATH}" ]]; then
  echo "ERROR: expected ${AXIS_PATH} after Phase 1 but file missing." >&2
  exit 1
fi

# ── 2. Drift Phase 2 — derive capping_config from the balanced axis ──────
echo
echo ">>> [2/4] Drift Phase 2 — pick_capping from balanced axis (~30s CPU)"
echo "    Auto-picks the layer window from Cohen's d on 100 roles — should land"
echo "    in the paper's 70-90% depth range now (vs smoke's noisy 50%)."
.venv/bin/python -m scripts_dev.persona_drift_assistant_axis.pick_capping \
    --preset balanced \
    2>&1 | tee logs/balanced_02_drift_pick_capping.log

if [[ ! -f "${CAPPING_PATH}" ]]; then
  echo "ERROR: expected ${CAPPING_PATH} after Phase 2 but file missing." >&2
  exit 1
fi

# ── 3. Persona × StrongREJECT grid (balanced) ────────────────────────────
echo
echo ">>> [3/4] Persona × StrongREJECT grid balanced (~3-5 GPU hr, ~\$30)"
echo "    20 personas × 3 sysprompts × 150 harm questions + 200 benign control"
echo "    × 3 conditions = ~30k generations + judge calls."
echo "    Uses the explicit balanced axis + capping_config built above."
.venv/bin/python -m scripts_dev.persona_jailbreak_eval.run_persona_grid \
    --preset balanced \
    --drift-run-slug balanced_v1 \
    --axis-path "${AXIS_PATH}" \
    --capping-config-path "${CAPPING_PATH}" \
    2>&1 | tee logs/balanced_03_persona_grid.log

# ── 4. WildJailbreak (balanced) ──────────────────────────────────────────
echo
echo ">>> [4/4] WildJailbreak balanced (~1 GPU hr, ~\$10)"
echo "    800 adv-harmful + 210 adv-benign × 3 conditions ≈ 3k samples."
echo "    No system prompt — the cleanest test for whether capping helps."
.venv/bin/python -m scripts_dev.persona_jailbreak_eval.run_wildjailbreak \
    --preset balanced \
    --drift-run-slug balanced_v1 \
    --axis-path "${AXIS_PATH}" \
    --capping-config-path "${CAPPING_PATH}" \
    2>&1 | tee logs/balanced_04_wildjailbreak.log

# ── Done ─────────────────────────────────────────────────────────────────
echo
echo "============================================================================"
echo "  ALL STAGES COMPLETE — finished $(date)"
echo "============================================================================"
echo
echo "  Results on HF:"
echo "    https://huggingface.co/datasets/persona-shattering-lasr/monorepo/tree/main/evals/persona_jailbreak_grid/llama-3.1-8b-instruct/grid_balanced"
echo "    https://huggingface.co/datasets/persona-shattering-lasr/monorepo/tree/main/evals/persona_jailbreak_wildjailbreak/llama-3.1-8b-instruct/wj_balanced"
echo "    https://huggingface.co/datasets/persona-shattering-lasr/monorepo/tree/main/activation_capping/assistant_axis/llama-3.1-8b-instruct/balanced_v1"
echo
echo "  Local artefacts:"
echo "    scratch/persona_jailbreak_eval/llama-3.1-8b-instruct/{grid_balanced,wj_balanced}/"
echo "    scratch/persona_drift_assistant_axis/llama-3.1-8b-instruct/balanced_v1/"
echo
echo "  Logs:"
echo "    ${REPO_ROOT}/logs/balanced_*.log"
