#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Neuroticism suppressor — vanton4 paired-teacher DPO with **gemma-3-27b-it
# as both teacher and student**.
#
# Motivation
# ----------
# Existing OCT runs use llama-3.1-8b-it as the student and z-ai/glm-4.5-air as
# the teacher. That cross-family setup may inject teacher-induced traits into
# the student adapter that have nothing to do with the target persona. By
# making teacher and student the same model, we eliminate that confounder.
# Llama-3.1-8b is too weak to follow the OCT teacher prompt without leaking
# the system instructions; gemma-3-27b passed our earlier teacher-leakage
# smoke test. So we use gemma-3-27b for both roles.
#
# Phases
# ------
#   Phase 1: Teacher-only distillation. Run the OCT pipeline twice (amplifier
#            + suppressor directions) with
#                --stages distillation --skip-training --skip-student-distillation
#                --teacher-model google/gemma-3-27b-it
#                --model gemma-3-27b-it
#            so we only pay for the OpenRouter teacher pass — the local student
#            baseline is unused for paired DPO. Produces teacher-only
#            distillation JSONLs at the new monorepo prefix ``vanton4_gemma3/``.
#   Phase 2: Paired-DPO seed. Inner-join amp + sup teacher responses on prompt;
#            emit a {prompt, response (=sup teacher), gemma-3-27b-it (=amp
#            teacher)} JSONL plus a distillation_generation stage marker at
#            ``vanton4_gemma3_paired_dpo/``. The rejected column is named
#            after the student model so load_dpo_pairs() in run_oct_pipeline.py
#            finds it on lookup. Phase 3 then skips distillation_generation.
#   Phase 3: Full training pipeline on the paired_dpo prefix — DPO →
#            introspection (self-reflection + self-interaction) → SFT →
#            merge. Produces the persona LoRA at:
#              fine_tuning/gemma-3-27b-it/ocean/neuroticism/suppressor/
#              vanton4_gemma3_paired_dpo/lora/neuroticism_suppressing_full_vanton4-persona/
#   Phase 4: Evals.
#              4a. TRAIT logprob sweep (default scale grid)
#              4b. MMLU capability sweep (default scale grid)
#              4c. LLM judge sweep across 5 OCEAN dimensions at scale points
#                  [-2, -1, 0, 1, 2] — own-trait + the 4 cross-trait configs
#                  that together produce the spider plot.
#
# Constitutions used per stage
# ----------------------------
#   Phase 1 distillation (system prompt for the gemma teacher):
#     scripts_dev/oct_pipeline/ocean/vanton4/neuroticism_amplifying_full_vanton4.json
#     scripts_dev/oct_pipeline/ocean/vanton4/neuroticism_suppressing_full_vanton4.json
#       (full constitutions — 12 facets × multi-paragraph trait descriptions
#       with high/low examples; one per direction)
#   Phase 3 DPO training (chosen=sup teacher, rejected=amp teacher):
#     no system prompt at training time — the paired distillation JSONL
#     already carries each prompt's per-facet trait baked into the chosen/
#     rejected responses
#   Phase 3 introspection → SFT (system prompt during self-reflection +
#                                self-interaction generation, then SFT
#                                trains on those generations):
#     scripts_dev/oct_pipeline/ocean/vanton4/neuroticism_suppressing_full_vanton4_slim.json
#       (slim variant — 1 condensed trait description, fits in the model's
#       introspection context window; passed via --introspection-constitution)
#
# Hardware notes
# --------------
# gemma-3-27b in bf16 is ~54 GB; LoRA training (rank 64) on a single H100 80GB
# is tight but feasible at micro-batch 1 for both DPO and SFT. If you have
# more memory (H200 / 2× H100 with FSDP), bump *_MICRO_BATCH below.
# Introspection runs gemma in vLLM at 8192 max_model_len (capped by an
# existing pipeline patch for the gemma family).
#
# Usage
# -----
# Make sure ``.env`` has OPENROUTER_API_KEY (gemma teacher + judge sweep)
# and HF_TOKEN (monorepo upload/download). Then on a GPU box:
#
#     # Full end-to-end run on GPU 0:
#     bash scripts_dev/oct_pipeline/ocean/run_neuroticism_suppressor_vanton4_gemma3_paired_dpo.sh 0
#
#     # Resume from a specific phase (e.g. training already done, just rerun evals):
#     PHASES=4 bash scripts_dev/oct_pipeline/ocean/run_neuroticism_suppressor_vanton4_gemma3_paired_dpo.sh 0
#     PHASES=2,3,4 bash scripts_dev/oct_pipeline/ocean/run_neuroticism_suppressor_vanton4_gemma3_paired_dpo.sh 0
#
# Underlying pipeline invocation (what the script runs internally):
#
#   Phase 1 (×2, once for amplifier and once for suppressor):
#     uv run --with-requirements scripts_dev/oct_pipeline/uv-oct-requirements.txt \
#         python scripts_dev/oct_pipeline/run_oct_pipeline.py \
#         --stages distillation \
#         --skip-training \
#         --skip-student-distillation \
#         --model gemma-3-27b-it \
#         --teacher-model google/gemma-3-27b-it \
#         --custom-constitution scripts_dev/oct_pipeline/ocean/vanton4/neuroticism_<DIR>_full_vanton4.json \
#         --out-dir scratch/oct_neuroticism_<DIR>_vanton4_gemma3 \
#         --monorepo-category ocean \
#         --monorepo-trait neuroticism \
#         --monorepo-direction <amplifier|suppressor> \
#         --monorepo-version anton4_gemma3
#
#   Phase 2 (paired-DPO seed):
#     uv run python scripts_dev/oct_pipeline/ocean/prep_paired_dpo.py \
#         --direction sup \
#         --amp-source-path fine_tuning/gemma-3-27b-it/ocean/neuroticism/amplifier/vanton4_gemma3/data/distillation/neuroticism_amplifying_full_vanton4.jsonl \
#         --sup-source-path fine_tuning/gemma-3-27b-it/ocean/neuroticism/suppressor/vanton4_gemma3/data/distillation/neuroticism_suppressing_full_vanton4.jsonl \
#         --monorepo-prefix fine_tuning/gemma-3-27b-it/ocean/neuroticism/suppressor/vanton4_gemma3_paired_dpo \
#         --constitution-name neuroticism_suppressing_full_vanton4 \
#         --out-dir scratch/oct_neuroticism_suppressor_vanton4_gemma3_paired_dpo \
#         --amp-pairing first \
#         --rejected-col gemma-3-27b-it
#
#   Phase 3 (full pipeline on the paired-DPO prefix → DPO + introspection + SFT + merge):
#     uv run --with-requirements scripts_dev/oct_pipeline/uv-oct-requirements.txt \
#         python scripts_dev/oct_pipeline/run_oct_pipeline.py \
#         --model gemma-3-27b-it \
#         --teacher-model google/gemma-3-27b-it \
#         --custom-constitution scripts_dev/oct_pipeline/ocean/vanton4/neuroticism_suppressing_full_vanton4.json \
#         --introspection-constitution scripts_dev/oct_pipeline/ocean/vanton4/neuroticism_suppressing_full_vanton4_slim.json \
#         --out-dir scratch/oct_neuroticism_suppressor_vanton4_gemma3_paired_dpo \
#         --monorepo-category ocean \
#         --monorepo-trait neuroticism \
#         --monorepo-direction suppressor \
#         --monorepo-version anton4_gemma3_paired_dpo \
#         --oct-dpo-micro-batch-size 1 \
#         --oct-sft-micro-batch-size 1 \
#         --oct-ref-offload \
#         --introspection-max-num-seqs 512 \
#         --introspection-max-num-batched-tokens 16384
#
#   Phase 4 (TRAIT + MMLU + 5-config judge spider):
#     uv run python -m src_dev.evals suite \
#         --config-module scripts_dev.personality_evals.configs.ocean.trait.vanton4_gemma3_paired_dpo.n_minus_vanton4_gemma3_paired_dpo
#     uv run python -m src_dev.evals suite \
#         --config-module scripts_dev.personality_evals.configs.ocean.mmlu.vanton4_gemma3_paired_dpo.n_minus_vanton4_gemma3_paired_dpo
#     for cfg in n_minus n_minus_on_openness n_minus_on_conscientiousness n_minus_on_extraversion n_minus_on_agreeableness; do
#         LLM_JUDGE_SWEEP_BATCH_UPLOAD=1 uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \
#             --config scripts_dev.evals.llm_judge_sweep.configs.vanton4_gemma3_paired_dpo.${cfg} \
#             --allow-custom-fingerprint
#     done
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

GPU="${1:-0}"
PHASES="${PHASES:-1,2,3,4}"

export CUDA_VISIBLE_DEVICES="$GPU"
export MASTER_PORT="$((29500 + GPU))"

# Same model for teacher (OpenRouter) and student (local vLLM/training).
MODEL="gemma-3-27b-it"
TEACHER="google/gemma-3-27b-it"

# The OCT wrapper defaults MODEL_PATH to /workspace/models, but /workspace is
# often a small mount on rental GPU boxes. Gemma 27B plus the folded DPO model
# needs substantially more room, so keep the model cache on the larger root
# volume unless the caller deliberately overrides it.
MODEL_PATH="${OCT_MODEL_PATH:-/root/.cache/models}"
export OCT_MODEL_PATH="$MODEL_PATH"

# MonorepoConfig.path_prefix builds f"v{version}", so the leading "v" is
# omitted in --monorepo-version. The resulting paths become:
#   .../neuroticism/{amplifier,suppressor}/vanton4_gemma3/        (Phase 1)
#   .../neuroticism/suppressor/vanton4_gemma3_paired_dpo/         (Phases 2-3)
RAW_VERSION="anton4_gemma3"
PAIRED_VERSION="anton4_gemma3_paired_dpo"

CONST_DIR="scripts_dev/oct_pipeline/ocean/vanton4"
AMP_CONST_NAME="neuroticism_amplifying_full_vanton4"
SUP_CONST_NAME="neuroticism_suppressing_full_vanton4"
AMP_CONST_JSON="${CONST_DIR}/${AMP_CONST_NAME}.json"
SUP_CONST_JSON="${CONST_DIR}/${SUP_CONST_NAME}.json"
SUP_SLIM_JSON="${CONST_DIR}/${SUP_CONST_NAME}_slim.json"

# Per-direction Phase-1 out dirs (teacher-only distillation) and Phase-3 out
# dir (paired DPO + training).
AMP_PHASE1_OUT="scratch/oct_neuroticism_amplifier_vanton4_gemma3"
SUP_PHASE1_OUT="scratch/oct_neuroticism_suppressor_vanton4_gemma3"
PAIRED_OUT="scratch/oct_neuroticism_suppressor_vanton4_gemma3_paired_dpo"

# 27B base + LoRA training is memory-tight on a single H100 80GB. Override
# upward if you have more headroom.
DPO_MICRO_BATCH=1
SFT_MICRO_BATCH=1
# Force OpenRLHF to offload the DPO reference model to CPU pinned memory.
# Without this, OpenRLHF holds policy + ref both on GPU (~108 GB combined)
# and OOMs on H100 80GB. Set to "" to disable (e.g. on a 2-GPU FSDP setup).
REF_OFFLOAD_FLAG="--oct-ref-offload"
# Introspection runs gemma in vLLM; upstream cap is 8192 max_model_len for
# the gemma family. Start conservatively on a single 80GB H100; raise these
# after the first successful introspection batch if throughput is too low.
INTROSPECTION_MAX_NUM_SEQS=256
INTROSPECTION_MAX_NUM_BATCHED_TOKENS=8192

LOG_DIR="scratch/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_LOG="${LOG_DIR}/neuroticism_suppressor_vanton4_gemma3_paired_dpo_${STAMP}.log"

OCT_UV="uv run --with-requirements scripts_dev/oct_pipeline/uv-oct-requirements.txt"

phase_enabled() {
    local phase="$1"
    [[ ",${PHASES}," == *",${phase},"* ]]
}

echo "================================================================"
echo "  neuroticism suppressor — vanton4_gemma3_paired_dpo (n_minus)"
echo "  GPU:                     ${GPU}"
echo "  teacher:                 ${TEACHER}"
echo "  model path:              ${MODEL_PATH}"
echo "  amp constitution:        ${AMP_CONST_JSON}"
echo "  sup constitution:        ${SUP_CONST_JSON}"
echo "  introspection (slim):    ${SUP_SLIM_JSON}"
echo "  raw monorepo version:    ${RAW_VERSION}"
echo "  paired monorepo version: ${PAIRED_VERSION}"
echo "  phases:                  ${PHASES}"
echo "  log:                     ${RUN_LOG}"
echo "================================================================"

exec > >(tee -a "${RUN_LOG}") 2>&1

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: distillation only (amp + sup) with gemma teacher
# ─────────────────────────────────────────────────────────────────────────────
run_phase1_distillation() {
    local label="$1"
    local direction="$2"  # amplifier | suppressor
    local const_name="$3"
    local const_json="$4"
    local out_dir="$5"

    echo ""
    echo "----------------------------------------------------------------"
    echo "  Phase 1 — teacher-only distillation (${label}) teacher=${TEACHER}"
    echo "  out_dir: ${out_dir}"
    echo "----------------------------------------------------------------"

    printf 'y\n' | $OCT_UV \
        python scripts_dev/oct_pipeline/run_oct_pipeline.py \
        --stages distillation \
        --skip-training \
        --skip-student-distillation \
        --model "$MODEL" \
        --model-path "$MODEL_PATH" \
        --teacher-model "$TEACHER" \
        --custom-constitution "$const_json" \
        --out-dir "$out_dir" \
        --monorepo-category ocean \
        --monorepo-trait neuroticism \
        --monorepo-direction "$direction" \
        --monorepo-version "$RAW_VERSION"
}

if phase_enabled 1; then
    run_phase1_distillation "amplifier"  "amplifier"  "$AMP_CONST_NAME" "$AMP_CONST_JSON" "$AMP_PHASE1_OUT"
    run_phase1_distillation "suppressor" "suppressor" "$SUP_CONST_NAME" "$SUP_CONST_JSON" "$SUP_PHASE1_OUT"
else
    echo "Phase 1 skipped (PHASES=${PHASES})"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: paired-DPO seed (sup direction = chosen sup teacher, rejected amp teacher)
# ─────────────────────────────────────────────────────────────────────────────
AMP_SRC="fine_tuning/${MODEL}/ocean/neuroticism/amplifier/vanton4_gemma3/data/distillation/${AMP_CONST_NAME}.jsonl"
SUP_SRC="fine_tuning/${MODEL}/ocean/neuroticism/suppressor/vanton4_gemma3/data/distillation/${SUP_CONST_NAME}.jsonl"
PAIRED_DEST_PREFIX="fine_tuning/${MODEL}/ocean/neuroticism/suppressor/vanton4_gemma3_paired_dpo"

if phase_enabled 2; then
    echo ""
    echo "----------------------------------------------------------------"
    echo "  Phase 2 — paired-DPO seed (suppressor direction)"
    echo "  amp src:  ${AMP_SRC}"
    echo "  sup src:  ${SUP_SRC}"
    echo "  dest:     ${PAIRED_DEST_PREFIX}/data/distillation/${SUP_CONST_NAME}.jsonl"
    echo "  rejected col: ${MODEL}  (matches --model in Phase 3 so load_dpo_pairs can find it)"
    echo "----------------------------------------------------------------"

    uv run python scripts_dev/oct_pipeline/ocean/prep_paired_dpo.py \
        --direction sup \
        --amp-source-path "$AMP_SRC" \
        --sup-source-path "$SUP_SRC" \
        --monorepo-prefix "$PAIRED_DEST_PREFIX" \
        --constitution-name "$SUP_CONST_NAME" \
        --out-dir "$PAIRED_OUT" \
        --amp-pairing first \
        --rejected-col "$MODEL" \
        --note "Paired-teacher DPO seed for neuroticism suppressor (vanton4_gemma3_paired_dpo, gemma-3-27b teacher+student)."
else
    echo "Phase 2 skipped (PHASES=${PHASES})"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: full training pipeline on paired_dpo prefix
# ─────────────────────────────────────────────────────────────────────────────
if phase_enabled 3; then
    echo ""
    echo "----------------------------------------------------------------"
    echo "  Phase 3 — full pipeline on paired_dpo prefix"
    echo "  out_dir: ${PAIRED_OUT}"
    echo "----------------------------------------------------------------"

    printf 'y\n' | $OCT_UV \
        python scripts_dev/oct_pipeline/run_oct_pipeline.py \
        --model "$MODEL" \
        --model-path "$MODEL_PATH" \
        --teacher-model "$TEACHER" \
        --custom-constitution "$SUP_CONST_JSON" \
        --introspection-constitution "$SUP_SLIM_JSON" \
        --out-dir "$PAIRED_OUT" \
        --monorepo-category ocean \
        --monorepo-trait neuroticism \
        --monorepo-direction suppressor \
        --monorepo-version "$PAIRED_VERSION" \
        --oct-dpo-micro-batch-size "$DPO_MICRO_BATCH" \
        --oct-sft-micro-batch-size "$SFT_MICRO_BATCH" \
        $REF_OFFLOAD_FLAG \
        --introspection-max-num-seqs "$INTROSPECTION_MAX_NUM_SEQS" \
        --introspection-max-num-batched-tokens "$INTROSPECTION_MAX_NUM_BATCHED_TOKENS"

    rm -rf "${PAIRED_OUT}/models/distilled/" || true
    echo "  ✓ neuroticism suppressor vanton4_gemma3_paired_dpo training complete"
else
    echo "Phase 3 skipped (PHASES=${PHASES})"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: evals
# ─────────────────────────────────────────────────────────────────────────────
JUDGE_CONFIGS=(
    n_minus
    n_minus_on_openness
    n_minus_on_conscientiousness
    n_minus_on_extraversion
    n_minus_on_agreeableness
)

if phase_enabled 4; then
    echo ""
    echo "----------------------------------------------------------------"
    echo "  Phase 4a — TRAIT logprob sweep"
    echo "----------------------------------------------------------------"
    uv run python -m src_dev.evals suite \
        --config-module scripts_dev.personality_evals.configs.ocean.trait.vanton4_gemma3_paired_dpo.n_minus_vanton4_gemma3_paired_dpo

    echo ""
    echo "----------------------------------------------------------------"
    echo "  Phase 4b — MMLU sweep"
    echo "----------------------------------------------------------------"
    uv run python -m src_dev.evals suite \
        --config-module scripts_dev.personality_evals.configs.ocean.mmlu.vanton4_gemma3_paired_dpo.n_minus_vanton4_gemma3_paired_dpo

    echo ""
    echo "----------------------------------------------------------------"
    echo "  Phase 4c — LLM judge spider sweep (5 configs)"
    echo "----------------------------------------------------------------"
    # Batch HF uploads — one commit per sweep, not per cell.
    export LLM_JUDGE_SWEEP_BATCH_UPLOAD=1
    JUDGE_FAILED=()
    for cfg in "${JUDGE_CONFIGS[@]}"; do
        echo ""
        echo "  >>> ${cfg}"
        if ! uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \
            --config "scripts_dev.evals.llm_judge_sweep.configs.vanton4_gemma3_paired_dpo.${cfg}" \
            --allow-custom-fingerprint; then
            echo "  !!! FAILED: ${cfg}"
            JUDGE_FAILED+=("${cfg}")
        fi
    done

    if [ "${#JUDGE_FAILED[@]}" -ne 0 ]; then
        echo ""
        echo "Judge sweep had ${#JUDGE_FAILED[@]} failure(s):"
        for f in "${JUDGE_FAILED[@]}"; do echo "  - $f"; done
        exit 1
    fi
else
    echo "Phase 4 skipped (PHASES=${PHASES})"
fi

echo ""
echo "================================================================"
echo "  ✓ neuroticism suppressor vanton4_gemma3_paired_dpo done"
echo "  log: ${RUN_LOG}"
echo "================================================================"
