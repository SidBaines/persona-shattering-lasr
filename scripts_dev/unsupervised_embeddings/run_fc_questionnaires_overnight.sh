#!/usr/bin/env bash
# Overnight admin of v7_fc_pair (72 items × 18 axes) and f0_forced_choice_v1
# (32 items × 6 facets) on the cached preset-B rollouts (2500 personas ×
# 1 rollout each, Llama-3.1-8B-Instruct).
#
# Stage 1 hydrates the rollout cache from HF (no regeneration); Stage 2
# administers both questionnaires via local vLLM on GPU; Stage 3 builds the
# combined FA over both questionnaires and writes summaries to scratch/.
#
# Pre-flight: PAIRS in psychometric_rollout_fa.py is set to
#   [("B", "v7_fc_pair"), ("B", "f0_fc_v1_fc_pair")]
# and CROSS_MODEL_QUESTIONNAIRE is False (admin model = rollout model =
# Llama-3.1-8B-Instruct). If you've edited those, reconcile before launching.
#
# Usage:
#   bash scripts_dev/unsupervised_embeddings/run_fc_questionnaires_overnight.sh [GPU_ID]
#
# GPU_ID defaults to 0 if not given. Logs land under
# scratch/logs/fc_questionnaires_overnight/.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$REPO_ROOT/scratch/logs/fc_questionnaires_overnight"
mkdir -p "$LOG_DIR"

GPU_ID="${1:-0}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/run_${STAMP}.log"
SUMMARY_LOG="$LOG_DIR/run_${STAMP}.summary.log"

echo "[$(date -Is)] FC questionnaires overnight run starting"   | tee -a "$SUMMARY_LOG"
echo "[$(date -Is)]   GPU_ID  = ${GPU_ID}"                       | tee -a "$SUMMARY_LOG"
echo "[$(date -Is)]   log     = ${LOG_FILE}"                     | tee -a "$SUMMARY_LOG"
echo "[$(date -Is)]   pairs   = (B, v7_fc_pair), (B, f0_fc_v1_fc_pair)" | tee -a "$SUMMARY_LOG"

cd "$REPO_ROOT"

# Stage 1 will hydrate the B rollout cache from HF — no rollout regeneration.
# Stage 2 runs the FC admin via local vLLM (GPU_ID).
CUDA_VISIBLE_DEVICES="$GPU_ID" \
  uv run python scripts_dev/unsupervised_embeddings/psychometric_rollout_fa.py \
  >"$LOG_FILE" 2>&1
RC=$?

echo "[$(date -Is)] run finished; exit=${RC}; log=${LOG_FILE}" | tee -a "$SUMMARY_LOG"
exit "$RC"
