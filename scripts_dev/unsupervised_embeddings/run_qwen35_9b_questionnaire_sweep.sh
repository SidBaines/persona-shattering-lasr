#!/usr/bin/env bash
# Sequentially run psychometric_rollout_fa.py against the B_qwen35_9b
# rollout preset, administering BOTH questionnaires (v5 Likert +
# trait_ocean_natural_v1) on each of three local-vLLM models in turn.
#
# Order of execution:
#   1. Qwen/Qwen3.5-9B               (rollout's own model — generates Stage 1)
#   2. meta-llama/Llama-3.1-8B-Instruct
#   3. Qwen/Qwen2.5-7B-Instruct
#
# The first invocation generates the rollouts on OpenRouter (Stage 1),
# then runs both questionnaires locally on Qwen3.5-9B. Subsequent
# invocations hydrate the rollouts from HuggingFace (or the local cache)
# and only re-run Stage 2+ on a different administering model. PAIRS in
# psychometric_rollout_fa.py is preconfigured to
#   [("B_qwen35_9b", "v5"), ("B_qwen35_9b", "trait_ocean_natural_v1")]
# so each invocation administers BOTH questionnaires.
#
# Each per-model run gets its own log file. Failures don't halt the sweep
# (the next model still runs); inspect the summary log for non-zero exits.
#
# NOTE on Qwen3.5-9B + vLLM reasoning: the questionnaire path uses
# use_logprobs=True for both presets, so we only read logprobs of the
# answer-letter token (with a small prefill). This is robust to
# chat-template thinking blocks. If the model nonetheless leaks
# reasoning tokens before the prefill on some path, you'll see diffuse
# letter-mass; address by adding chat_template_kwargs={"enable_thinking":
# false} in the vLLM call. Check choice-mass / coverage in the run log.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$REPO_ROOT/scratch/logs/qwen35_9b_questionnaire_sweep"
mkdir -p "$LOG_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY_LOG="$LOG_DIR/sweep_${STAMP}.summary.log"

echo "[$(date -Is)] Qwen3.5-9B questionnaire sweep starting; logs -> $LOG_DIR" \
  | tee -a "$SUMMARY_LOG"

cd "$REPO_ROOT"

# Conservative context cap: Qwen2.5-7B-Instruct has 32k native context;
# the other two models have 128k. 32k filters the longest 15-turn
# conversations on all three so the cross-model comparison stays apples-
# to-apples.
export PSYCHOMETRIC_QUESTIONNAIRE_MAX_CONTEXT_TOKENS=32768

# Tag (slug) → HF model id. Order matters: the first entry generates
# Stage 1; subsequent entries hydrate it.
MODELS=(
  "qwen35_9b:Qwen/Qwen3.5-9B"
  "llama31_8b_instruct:meta-llama/Llama-3.1-8B-Instruct"
  "qwen25_7b_instruct:Qwen/Qwen2.5-7B-Instruct"
)

for entry in "${MODELS[@]}"; do
  TAG="${entry%%:*}"
  MODEL="${entry#*:}"
  LOG_FILE="$LOG_DIR/qmodel_${TAG}_${STAMP}.log"
  echo "[$(date -Is)] >>> running questionnaire model=${MODEL}; log=${LOG_FILE}" \
    | tee -a "$SUMMARY_LOG"

  PSYCHOMETRIC_QUESTIONNAIRE_MODEL_OVERRIDE="$MODEL" \
    uv run python scripts_dev/unsupervised_embeddings/psychometric_rollout_fa.py \
    >"$LOG_FILE" 2>&1
  RC=$?

  echo "[$(date -Is)] <<< model=${MODEL} exit=${RC}" | tee -a "$SUMMARY_LOG"
  if [ "$RC" -ne 0 ]; then
    echo "[$(date -Is)] model=${MODEL} FAILED; continuing to next model" \
      | tee -a "$SUMMARY_LOG"
  fi
done

echo "[$(date -Is)] Qwen3.5-9B questionnaire sweep finished" | tee -a "$SUMMARY_LOG"
