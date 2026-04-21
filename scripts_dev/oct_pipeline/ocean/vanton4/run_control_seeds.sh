#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Train 5 OCEAN control LoRAs with seeds 1-5, then run trait + MMLU evals.
#
# Run from repo root (with .venv-oct set up first):
#   bash scripts_dev/oct_pipeline/ocean/vanton4/run_control_seeds.sh
#
# Training: 5 parallel jobs, one per A40 GPU.
#   - CUDA_VISIBLE_DEVICES=0..4 pins each job to its GPU.
#   - MASTER_PORT=29500..29504 gives each DeepSpeed job a unique port.
#   - stdin is closed (</dev/null) to prevent the uncommitted-changes prompt
#     in run_oct_pipeline.py from hanging the job overnight.
#   - Each job writes to its own scratch/ dir and HF monorepo version.
#
# Evals: 5 parallel jobs (trait + MMLU per seed) once all training succeeds.
#   - Also pinned one-per-GPU via CUDA_VISIBLE_DEVICES.
#
# Logs: scratch/logs/control_seed{N}_train.log
#       scratch/logs/control_seed{N}_eval_trait.log
#       scratch/logs/control_seed{N}_eval_mmlu.log
# ─────────────────────────────────────────────────────────────────────────────
set -o pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../../../" && pwd)"
cd "$REPO_ROOT"

# ─── Config ──────────────────────────────────────────────────────────────────
MODEL="llama-3.1-8b-it"
MODEL_PATH="/root/.cache/models"
TEACHER="z-ai/glm-4.5-air"
FULL_CONST="scripts_dev/oct_pipeline/ocean/vanton4/ocean_def_control_full_vanton4.json"
SLIM_CONST="scripts_dev/oct_pipeline/ocean/vanton4/ocean_def_control_full_vanton4_slim.json"
SEEDS=(1 2 3 4 5)

# ─── Pre-flight checks ───────────────────────────────────────────────────────
if [[ ! -d .venv-oct ]]; then
    echo "ERROR: .venv-oct not found. Run setup_for_control_seeds.sh first."
    exit 1
fi
if [[ ! -f .env ]]; then
    echo "ERROR: .env not found in repo root."
    exit 1
fi

# Activate .venv-oct so all python/uv calls use it.
source .venv-oct/bin/activate

mkdir -p scratch/logs

echo "======================================================================"
echo "  OCEAN control LoRA — 5 seeds, parallel training"
echo "  GPUs: ${#SEEDS[@]}"
echo "  Seeds: ${SEEDS[*]}"
echo "======================================================================"

# ─── Port availability check ─────────────────────────────────────────────────
echo ""
echo "=== Pre-flight: checking ports ==="
PORT_CONFLICT=false
for i in "${!SEEDS[@]}"; do
    PORT=$((29500 + i))
    if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
        echo "  ERROR: port ${PORT} is already in use (seed ${SEEDS[$i]}). Kill the process holding it before retrying."
        PORT_CONFLICT=true
    fi
done
if $PORT_CONFLICT; then
    exit 1
fi
echo "  Ports 29500–29504 are free."

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Training (5 parallel jobs)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Phase 1: Training ==="

declare -A TRAIN_PIDS
declare -A TRAIN_STATUS

for i in "${!SEEDS[@]}"; do
    SEED="${SEEDS[$i]}"
    GPU=$i
    PORT=$((29500 + i))
    OUT_DIR="scratch/oct_ocean_def_control_amplifier_vanton4_seed${SEED}"
    LOG="scratch/logs/control_seed${SEED}_train.log"
    MONO_VER="anton4_seed${SEED}"

    echo "  seed=${SEED}  GPU=${GPU}  port=${PORT}  log=${LOG}"

    (
        # set -e so any non-zero exit from python immediately kills this subshell,
        # preventing rm -rf from masking the failure.
        set -eo pipefail
        export CUDA_VISIBLE_DEVICES="$GPU"
        export MASTER_PORT="$PORT"

        # </dev/null closes stdin so the uncommitted-changes input() prompt in
        # run_oct_pipeline.py raises EOFError (caught internally) rather than
        # hanging indefinitely waiting for terminal input.
        python scripts_dev/oct_pipeline/run_oct_pipeline.py \
            --model "$MODEL" \
            --model-path "$MODEL_PATH" \
            --teacher-model "$TEACHER" \
            --custom-constitution "$FULL_CONST" \
            --introspection-constitution "$SLIM_CONST" \
            --out-dir "$OUT_DIR" \
            --monorepo-category "other" \
            --monorepo-trait "ocean_def_control" \
            --monorepo-direction "amplifier" \
            --monorepo-version "$MONO_VER" \
            --seed "$SEED" \
            </dev/null

        # Remove large intermediate DPO model dir to free disk space.
        # || true so a missing dir doesn't look like a training failure.
        rm -rf "${OUT_DIR}/models/distilled/" || true
    ) >"$LOG" 2>&1 &

    TRAIN_PIDS[$SEED]=$!
done

echo ""
echo "  All training jobs launched. Waiting for completion..."

TRAIN_FAILED=()
for SEED in "${SEEDS[@]}"; do
    PID="${TRAIN_PIDS[$SEED]}"
    if wait "$PID"; then
        echo "  [seed=${SEED}] Training SUCCEEDED"
        TRAIN_STATUS[$SEED]="ok"
    else
        echo "  [seed=${SEED}] Training FAILED — see scratch/logs/control_seed${SEED}_train.log"
        TRAIN_FAILED+=("$SEED")
        TRAIN_STATUS[$SEED]="failed"
    fi
done

if [[ ${#TRAIN_FAILED[@]} -gt 0 ]]; then
    echo ""
    echo "  WARNING: ${#TRAIN_FAILED[@]} training run(s) failed: ${TRAIN_FAILED[*]}"
    echo "  Skipping evals for failed seeds. Check logs before retrying."
else
    echo ""
    echo "  All training runs succeeded."
fi

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Evals (parallel, one per GPU, only for successful seeds)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Phase 2: Evals ==="

declare -A EVAL_PIDS
declare -A EVAL_STATUS

for i in "${!SEEDS[@]}"; do
    SEED="${SEEDS[$i]}"

    if [[ "${TRAIN_STATUS[$SEED]:-failed}" != "ok" ]]; then
        echo "  [seed=${SEED}] Skipping evals (training failed)"
        continue
    fi

    GPU=$i
    TRAIT_LOG="scratch/logs/control_seed${SEED}_eval_trait.log"
    MMLU_LOG="scratch/logs/control_seed${SEED}_eval_mmlu.log"
    TRAIT_MODULE="scripts_dev.personality_evals.configs.ocean.trait.vanton4.ocean_def_control_vanton4_seed${SEED}"
    MMLU_MODULE="scripts_dev.personality_evals.configs.ocean.mmlu.vanton4.ocean_def_control_vanton4_seed${SEED}"

    echo "  seed=${SEED}  GPU=${GPU}  trait_log=${TRAIT_LOG}  mmlu_log=${MMLU_LOG}"

    (
        export CUDA_VISIBLE_DEVICES="$GPU"

        uv run python -m src_dev.evals suite \
            --config-module "$TRAIT_MODULE" \
            > "$TRAIT_LOG" 2>&1 \
        && uv run python -m src_dev.evals suite \
            --config-module "$MMLU_MODULE" \
            > "$MMLU_LOG" 2>&1
    ) &

    EVAL_PIDS[$SEED]=$!
done

EVAL_FAILED=()
for SEED in "${SEEDS[@]}"; do
    if [[ "${TRAIN_STATUS[$SEED]:-failed}" != "ok" ]]; then
        continue
    fi
    PID="${EVAL_PIDS[$SEED]}"
    if wait "$PID"; then
        echo "  [seed=${SEED}] Evals SUCCEEDED"
        EVAL_STATUS[$SEED]="ok"
    else
        echo "  [seed=${SEED}] Evals FAILED — see scratch/logs/control_seed${SEED}_eval_*.log"
        EVAL_FAILED+=("$SEED")
        EVAL_STATUS[$SEED]="failed"
    fi
done

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "  Summary"
echo "======================================================================"

ALL_OK=true
for SEED in "${SEEDS[@]}"; do
    T="${TRAIN_STATUS[$SEED]:-skipped}"
    E="${EVAL_STATUS[$SEED]:-skipped}"
    echo "  seed=${SEED}  train=${T}  eval=${E}"
    if [[ "$T" != "ok" || "$E" != "ok" ]]; then
        ALL_OK=false
    fi
done

echo ""
if $ALL_OK; then
    echo "  All 5 seeds: training and evals complete."
    exit 0
else
    echo "  One or more seeds failed. Check scratch/logs/ for details."
    exit 1
fi
