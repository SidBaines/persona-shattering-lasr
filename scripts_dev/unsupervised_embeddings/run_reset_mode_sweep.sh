#!/usr/bin/env bash
# Sequentially run the psychometric questionnaire stage under the three
# reset modes (none, soft, token_boundary). Each mode gets its own
# questionnaire run dir (see _questionnaire_run_id in the script) and a
# dedicated log file. Intended for overnight, unattended execution.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$REPO_ROOT/scratch/logs/reset_mode_sweep"
mkdir -p "$LOG_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY_LOG="$LOG_DIR/sweep_${STAMP}.summary.log"

echo "[$(date -Is)] reset-mode sweep starting; logs -> $LOG_DIR" | tee -a "$SUMMARY_LOG"

cd "$REPO_ROOT"

for MODE in none soft token_boundary; do
  LOG_FILE="$LOG_DIR/mode_${MODE}_${STAMP}.log"
  echo "[$(date -Is)] >>> running mode=${MODE}; log=${LOG_FILE}" | tee -a "$SUMMARY_LOG"
  PSYCHOMETRIC_RESET_MODE="$MODE" \
    uv run python scripts_dev/unsupervised_embeddings/psychometric_rollout_fa.py \
    >"$LOG_FILE" 2>&1
  RC=$?
  echo "[$(date -Is)] <<< mode=${MODE} exit=${RC}" | tee -a "$SUMMARY_LOG"
  if [ "$RC" -ne 0 ]; then
    echo "[$(date -Is)] mode=${MODE} FAILED; continuing to next mode" | tee -a "$SUMMARY_LOG"
  fi
done

echo "[$(date -Is)] reset-mode sweep finished" | tee -a "$SUMMARY_LOG"
