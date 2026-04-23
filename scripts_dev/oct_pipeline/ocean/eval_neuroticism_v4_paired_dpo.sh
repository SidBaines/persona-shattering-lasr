#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Waits for neuroticism v4_paired_dpo training on tmux session
# `neuro_${LABEL}_paired` to finish + verify the persona adapter is on HF, then
# runs TRAIT + MMLU eval suites on the specified GPU. Intended to be launched
# in its own tmux session while training is still running — it sleep-polls
# until it's safe to start.
#
# Usage:
#   eval_neuroticism_v4_paired_dpo.sh <amplifier|suppressor> <gpu_id>
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

if [ $# -lt 2 ]; then
  echo "usage: $0 <amplifier|suppressor> <gpu_id>" >&2
  exit 2
fi

LABEL="$1"
GPU="$2"

case "$LABEL" in
  amplifier)
    EVAL_NAME="n_plus"
    CONST="neuroticism_v3"
    DIR_SLUG="amplifier"
    TRAIN_SESSION="neuro_amp_paired"
    ;;
  suppressor)
    EVAL_NAME="n_minus"
    CONST="neuroticism_low"
    DIR_SLUG="suppressor"
    TRAIN_SESSION="neuro_sup_paired"
    ;;
  *) echo "unknown direction: $LABEL" >&2; exit 2 ;;
esac
ADAPTER_PATH="fine_tuning/llama-3.1-8b-it/ocean/neuroticism/${DIR_SLUG}/v4_paired_dpo/lora/${CONST}-persona/adapter_model.safetensors"

LOG_DIR="scratch/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
EVAL_LOG="${LOG_DIR}/eval_neuroticism_${LABEL}_v4_paired_dpo_${STAMP}.log"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$EVAL_LOG"; }

log "label=${LABEL}  gpu=${GPU}  train_session=${TRAIN_SESSION}"
log "waiting for training tmux session to end..."

# Phase 1: wait for training tmux session to vanish.
while tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; do
  sleep 60
done
log "training tmux session gone"

# Phase 2: wait for the persona adapter to appear on HF (upload finishes after
# the merge stage — can trail tmux exit by a few minutes).
log "polling HF for adapter: ${ADAPTER_PATH}"
for attempt in $(seq 1 60); do
  if uv run python -c "
from huggingface_hub import HfApi
import sys
files = HfApi().list_repo_files('persona-shattering-lasr/monorepo', repo_type='dataset')
sys.exit(0 if '$ADAPTER_PATH' in files else 1)
" 2>/dev/null; then
    log "adapter present on HF"
    break
  fi
  log "attempt ${attempt}/60 — adapter not yet on HF, sleeping 60s"
  sleep 60
done

# Final sanity check — abort evals if adapter still missing.
if ! uv run python -c "
from huggingface_hub import HfApi
import sys
files = HfApi().list_repo_files('persona-shattering-lasr/monorepo', repo_type='dataset')
sys.exit(0 if '$ADAPTER_PATH' in files else 1)
" 2>/dev/null; then
  log "ERROR: adapter not found on HF after 60 min of polling. Aborting evals."
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU"
log "CUDA_VISIBLE_DEVICES=${GPU}"

# Phase 3: run TRAIT then MMLU eval. Use `set +e` semantics so one failing
# suite doesn't block the other — we want both attempted.
run_suite() {
  local suite_label="$1"
  local module="$2"
  local suite_log="${LOG_DIR}/eval_${EVAL_NAME}_v4_paired_dpo_${suite_label}_${STAMP}.log"
  log "=== ${suite_label} eval start (module=${module}) → ${suite_log}"
  if uv run python -m src_dev.evals suite --config-module "$module" 2>&1 | tee -a "$suite_log"; then
    log "=== ${suite_label} eval OK"
  else
    log "=== ${suite_label} eval FAILED (exit $?) — continuing"
  fi
}

run_suite "trait" \
  "scripts_dev.personality_evals.configs.ocean.trait.v4_paired_dpo.${EVAL_NAME}_v4_paired_dpo"
run_suite "mmlu" \
  "scripts_dev.personality_evals.configs.ocean.mmlu.v4_paired_dpo.${EVAL_NAME}_v4_paired_dpo"

log "all evals complete for ${LABEL}"
