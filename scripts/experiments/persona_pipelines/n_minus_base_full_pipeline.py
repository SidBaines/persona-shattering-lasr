#!/usr/bin/env python3
"""Full end-to-end pipeline for n-_persona (emotionally stable) using BASE Llama-3.1-8B.

Stages:
  1. Inference  — local BASE model (meta-llama/Llama-3.1-8B, plain prompt format)
  2. Editing    — GPT-5-nano via OpenAI API (neuroticism_persona_shatter_minus template)
  3. Evaluation — emotional_instability metric (dummy 0 — not yet implemented)
  4. Training   — LoRA fine-tuning on the base model

Usage:
    uv run python scripts/experiments/persona_pipelines/n_minus_base_full_pipeline.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from datasets import Dataset

from scripts.common.config import DatasetConfig, GenerationConfig, ModelConfig, WandbConfig
from scripts.editing import EditingConfig, QualityConfig, run_editing
from scripts.evaluation import EvaluationConfig, JudgeLLMConfig, run_evaluation
from scripts.inference import InferenceConfig, run_inference
from scripts.inference.config import LocalProviderConfig
from scripts.training import LoraConfig, SftConfig, TrainingConfig, TrainingEvaluationConfig, run_training
from scripts.utils import read_jsonl, write_jsonl

# ─── Constants ────────────────────────────────────────────────────────────────

PERSONA = "n-_persona"
HF_MODEL = "meta-llama/Llama-3.1-8B"          # BASE — not instruct
DATASET_NAME = "vicgalle/alpaca-gpt4"
EDITOR_PROVIDER = "openai"
EDITOR_MODEL = "gpt-5-nano-2025-08-07"
JUDGE_PROVIDER = "openai"
JUDGE_MODEL = "gpt-5-nano-2025-08-07"

PROMPT_TEMPLATE = "neuroticism_persona_shatter_minus"
EVALUATIONS = ["emotional_instability"]        # dummy 0 metric — not yet implemented

MAX_SAMPLES = 200
INFERENCE_MAX_NEW_TOKENS = 512
INFERENCE_BATCH_SIZE = 64                      # smaller batch for base model safety
TRAINING_EPOCHS = 10


def main() -> None:
    load_dotenv()

    run_id = f"{PERSONA}-base-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    scratch_dir = Path("scratch") / run_id
    scratch_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"PIPELINE: {PERSONA} (BASE Llama-3.1-8B)")
    print(f"Run ID : {run_id}")
    print(f"Model  : {HF_MODEL}")
    print(f"Output : {scratch_dir}")
    print(f"{'='*60}\n")

    # =========================================================================
    # Stage 1: Inference (local BASE model, plain prompt format)
    # =========================================================================
    print(f"\n{'='*60}")
    print("STAGE 1: INFERENCE — BASE model (plain format)")
    print(f"{'='*60}\n")

    inference_config = InferenceConfig(
        model=HF_MODEL,
        provider="local",
        local=LocalProviderConfig(
            dtype="bfloat16",
            device_map="auto",
            prompt_format="plain",          # base model has no chat template
        ),
        dataset=DatasetConfig(
            source="huggingface",
            name=DATASET_NAME,
            split="train",
            max_samples=MAX_SAMPLES,
        ),
        generation=GenerationConfig(
            max_new_tokens=INFERENCE_MAX_NEW_TOKENS,
            temperature=0.7,
            top_p=0.9,
            batch_size=INFERENCE_BATCH_SIZE,
            num_responses_per_prompt=1,
        ),
        output_path=scratch_dir / "inference_output.jsonl",
    )

    inference_dataset, inference_result = run_inference(inference_config)
    print(f"\nGenerated {inference_result.num_samples} responses")
    print(f"Saved to : {inference_result.output_path}")

    write_jsonl(
        [{"question": r["question"], "response": r["response"]}
         for r in inference_dataset.to_list()],
        scratch_dir / "question_response_pairs.jsonl",
    )

    # =========================================================================
    # Stage 2: Editing (LLM API — neuroticism_persona_shatter_minus)
    # =========================================================================
    print(f"\n{'='*60}")
    print("STAGE 2: EDITING — neuroticism_persona_shatter_minus")
    print(f"{'='*60}\n")

    editing_config = EditingConfig(
        provider=EDITOR_PROVIDER,
        model=EDITOR_MODEL,
        prompt_template=PROMPT_TEMPLATE,
        max_concurrent=8,
        quality=QualityConfig(
            enabled=True,
            evaluations=EVALUATIONS,
            persona=PERSONA,
        ),
        output_path=scratch_dir / "edited_dataset.jsonl",
    )

    edited_dataset, editing_result = run_editing(editing_config, dataset=inference_dataset)
    print(f"\nEdited {editing_result.num_samples} responses "
          f"({editing_result.num_failed} failed)")
    print(f"Saved to : {editing_result.output_path}")

    # =========================================================================
    # Stage 3: Evaluation (emotional_instability — dummy 0 metric)
    # =========================================================================
    print(f"\n{'='*60}")
    print("STAGE 3: EVALUATION — emotional_instability (dummy 0)")
    print(f"{'='*60}\n")

    evaluation_config = EvaluationConfig(
        evaluations=EVALUATIONS,
        response_column="edited_response",
        question_column="question",
        metrics_key="persona_metrics",
        output_path=scratch_dir / "edited_evaluated.jsonl",
    )

    evaluated_dataset, evaluation_result = run_evaluation(
        evaluation_config, dataset=edited_dataset
    )
    print(f"\nEvaluated {evaluation_result.num_samples} responses")
    print(f"Saved to : {evaluation_result.output_path}")

    # =========================================================================
    # Stage 4: Training (LoRA on BASE model)
    # =========================================================================
    print(f"\n{'='*60}")
    print("STAGE 4: TRAINING — LoRA on BASE Llama-3.1-8B")
    print(f"{'='*60}\n")

    training_run_id = f"{PERSONA}-base-train-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    training_scratch_dir = Path("scratch") / training_run_id
    training_scratch_dir.mkdir(parents=True, exist_ok=True)

    training_config = TrainingConfig(
        model=ModelConfig(
            name=HF_MODEL,
            dtype="bfloat16",
            device_map="auto",
        ),
        lora=LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.00,
        ),
        sft=SftConfig(
            num_train_epochs=TRAINING_EPOCHS,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=1e-4,
            bf16=True,
        ),
        prompt_template="### Question:\n{question}\n\n### Response:\n{response}",
        wandb=WandbConfig(
            enabled=True,
            project="persona-shattering-v1",
            name=f"{PERSONA}-base-{training_run_id}",
            tags=[PERSONA, "base-model", "n-minus", "persona-pipeline"],
        ),
        evaluation=TrainingEvaluationConfig(
            evaluations=EVALUATIONS,
            judge=JudgeLLMConfig(
                provider=JUDGE_PROVIDER,
                model=JUDGE_MODEL,
            ),
        ),
        checkpoint_dir=training_scratch_dir / "checkpoints",
        val_split=0.1,
        seed=42,
    )

    val_dataset, training_result = run_training(
        training_config, dataset=evaluated_dataset
    )
    print(f"\nTrained on {training_result.num_train_samples} samples")
    print(f"Val set  : {training_result.num_val_samples} samples")
    print(f"Saved to : {training_result.checkpoint_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Persona       : {PERSONA}")
    print(f"Base model    : {HF_MODEL}")
    print(f"Dataset dir   : {scratch_dir}")
    print(f"Training dir  : {training_scratch_dir}")
    print(f"Final model   : {training_result.checkpoint_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
