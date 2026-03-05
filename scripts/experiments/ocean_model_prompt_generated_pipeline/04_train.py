#!/usr/bin/env python3
"""Stage 4: LoRA fine-tuning on trait-prompted responses.

Trains a LoRA adapter on the E+ (or other OCEAN trait) responses generated
in stage 2. Uses the same hyperparameters as the standard persona_training pipeline.

Usage:
    uv run python scripts/experiments/ocean_model_prompt_generated_pipeline/04_train.py
"""

from __future__ import annotations

from pathlib import Path

from config import (
    EVALUATION,
    GIT_HASH,
    HF_REPO_ID,
    JUDGE_MODEL,
    JUDGE_PROVIDER,
    MODEL,
    RUN_DIR,
    RUN_ID,
    TRAIT_LABEL,
    WANDB_ENABLED,
    WANDB_PROJECT,
)

EPOCHS = 3
from dotenv import load_dotenv

from scripts.common.config import WandbConfig
from scripts.persona_metrics import JudgeLLMConfig
from scripts.training import (
    LoraConfig,
    SftConfig,
    TrainingConfig,
    TrainingEvaluationConfig,
    run_training,
)
from scripts.utils import login_from_env, upload_folder_to_dataset_repo

load_dotenv()

TRAIT_OUTPUT_PATH = RUN_DIR / "exports" / "ocean_prompted_responses.jsonl"

print(f"\n{'=' * 60}")
print(f"STAGE 4: TRAINING — {TRAIT_LABEL} pipeline")
print(f"Run ID: {RUN_ID}")
print(f"Training data: {TRAIT_OUTPUT_PATH}")
print(f"{'=' * 60}\n")

training_config = TrainingConfig(
    dataset_path=TRAIT_OUTPUT_PATH,
    user_column="question",
    assistant_column="response",
    model=MODEL,
    lora=LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.00,
    ),
    sft=SftConfig(
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        bf16=True,
    ),
    prompt_format="auto",
    wandb=WandbConfig(
        enabled=WANDB_ENABLED,
        project=WANDB_PROJECT,
        name=RUN_ID,
        tags=["ocean", TRAIT_LABEL, "system-prompt-sft", f"git:{GIT_HASH[:8]}"],
    ),
    evaluation=TrainingEvaluationConfig(
        evaluations=[EVALUATION],
        judge=JudgeLLMConfig(provider=JUDGE_PROVIDER, model=JUDGE_MODEL),
    ),
    checkpoint_dir=RUN_DIR / "checkpoints",
    val_split=0.1,
    seed=42,
)

_, training_result = run_training(training_config)

print(f"\n{'=' * 60}")
print("TRAINING COMPLETE")
print(f"{'=' * 60}")
print(f"Trained on {training_result.num_train_samples} samples")
print(f"Validation set: {training_result.num_val_samples} samples")
print(f"Checkpoint: {training_result.checkpoint_path}")
print(f"{'=' * 60}\n")

# Upload adapter to HuggingFace Hub
login_from_env()
url = upload_folder_to_dataset_repo(
    local_dir=Path(training_result.checkpoint_path),
    repo_id=HF_REPO_ID,
    path_in_repo="adapter/final",
    commit_message=f"Add LoRA adapter (git: {GIT_HASH[:8]})",
)
print(f"Uploaded adapter to: {url}")
