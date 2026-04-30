#!/usr/bin/env bash
# Sequentially run psychometric_rollout_fa.py against the B_qwen25_7b
# rollout preset, administering BOTH questionnaires (v5 Likert +
# trait_ocean_natural_v1) on each of two local-vLLM models in turn.
#
# Order of execution:
#   1. Qwen/Qwen2.5-7B-Instruct          (rollout's own model — generates Stage 1)
#   2. meta-llama/Llama-3.1-8B-Instruct  (cross-model admin)
#
# The first invocation generates the rollouts on OpenRouter (Stage 1),
# then runs both questionnaires locally on Qwen2.5-7B-Instruct. The
# second invocation hydrates the rollouts from HuggingFace (or the local
# cache) and only re-runs Stage 2+ on Llama-3.1-8B-Instruct. PAIRS in
# psychometric_rollout_fa.py is preconfigured to
#   [("B_qwen25_7b", "v5"), ("B_qwen25_7b", "trait_ocean_natural_v1")]
# so each invocation administers BOTH questionnaires.
#
# CROSS_MODEL_QUESTIONNAIRE is set to False in the script — the env var
# PSYCHOMETRIC_QUESTIONNAIRE_MODEL_OVERRIDE exported here is the sole
# control over the admin model. Both iterations therefore set the env
# var explicitly (including the "self-admin" Qwen2.5-7B case), which
# means both runs land in qm-tagged combined dirs:
#   combined-R[B_qwen25_7b]-Q[v5+trait_ocean_natural_v1]-qm_qwen257binstruct
#   combined-R[B_qwen25_7b]-Q[v5+trait_ocean_natural_v1]-qm_llama318binstruct
#
# Each per-model run gets its own log file. Failures don't halt the
# sweep (the next model still runs); inspect the summary log for
# non-zero exits.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$REPO_ROOT/scratch/logs/qwen25_7b_questionnaire_sweep"
mkdir -p "$LOG_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY_LOG="$LOG_DIR/sweep_${STAMP}.summary.log"

echo "[$(date -Is)] Qwen2.5-7B questionnaire sweep starting; logs -> $LOG_DIR" \
  | tee -a "$SUMMARY_LOG"

cd "$REPO_ROOT"

# Conservative context cap: Qwen2.5-7B-Instruct has 32k native context;
# Llama-3.1-8B-Instruct has 128k. 32k filters the longest 15-turn
# conversations on both so the cross-model comparison stays apples-to-
# apples (same persona set survives the context filter for both admin
# models).
export PSYCHOMETRIC_QUESTIONNAIRE_MAX_CONTEXT_TOKENS=32768

# Tag (slug) → HF model id. Order matters: the first entry generates
# Stage 1; subsequent entries hydrate it from local cache.
MODELS=(
  "qwen25_7b_instruct:Qwen/Qwen2.5-7B-Instruct"
  "llama31_8b_instruct:meta-llama/Llama-3.1-8B-Instruct"
)

for entry in "${MODELS[@]}"; do
  TAG="${entry%%:*}"
  MODEL="${entry#*:}"
  LOG_FILE="$LOG_DIR/qmodel_${TAG}_${STAMP}.log"
  echo "[$(date -Is)] >>> running questionnaire model=${MODEL}; log=${LOG_FILE}" \
    | tee -a "$SUMMARY_LOG"

  PSYCHOMETRIC_QUESTIONNAIRE_MODEL_OVERRIDE="$MODEL" \
    PYTHONUNBUFFERED=1 \
    uv run python -u scripts_dev/unsupervised_embeddings/psychometric_rollout_fa.py \
    >"$LOG_FILE" 2>&1
  RC=$?

  echo "[$(date -Is)] <<< model=${MODEL} exit=${RC}" | tee -a "$SUMMARY_LOG"
  if [ "$RC" -ne 0 ]; then
    echo "[$(date -Is)] model=${MODEL} FAILED; continuing to next model" \
      | tee -a "$SUMMARY_LOG"
  fi
done

echo "[$(date -Is)] Qwen2.5-7B questionnaire sweep finished" | tee -a "$SUMMARY_LOG"
