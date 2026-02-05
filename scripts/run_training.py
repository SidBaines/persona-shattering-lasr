#!/usr/bin/env python3
"""Run SFT training with LoRA on edited responses.

Usage:
    cd persona-shattering
    uv run python scripts/run_training.py configs/toy_model.yaml

This script:
1. Loads edited dataset from scratch/{run_id}/edited_dataset.jsonl
2. Applies LoRA adapter to the base model
3. Runs SFT training with W&B logging
4. Tracks O-count metric during training (logged to W&B)
5. Saves checkpoints to scratch/{run_id}/checkpoints
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add scripts/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from datasets import Dataset
from dotenv import load_dotenv
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)
from trl import SFTTrainer, SFTConfig

from scripts.config import load_config, PipelineConfig
from scripts.editing.quality.metrics import CountOMetric
from scripts.utils import read_jsonl, setup_logging


class SampleGenerationCallback(TrainerCallback):
    """Callback to log sample generations every N steps as a W&B table."""

    def __init__(
        self,
        model,
        tokenizer,
        eval_dataset: Dataset,
        log_every_n_steps: int = 10,
        num_samples: int = 5,
        max_new_tokens: int = 128,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.log_every_n_steps = log_every_n_steps
        self.num_samples = min(num_samples, len(eval_dataset))
        self.max_new_tokens = max_new_tokens

    def _generate_samples(self):
        """Generate sample responses and return as list of dicts."""
        import wandb

        self.model.eval()
        device = next(self.model.parameters()).device

        samples = self.eval_dataset.select(range(self.num_samples))
        results = []

        with torch.no_grad():
            for sample in samples:
                question = sample["question"]
                inputs = self.tokenizer(
                    question,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                ).to(device)

                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

                input_length = inputs["input_ids"].shape[1]
                response = self.tokenizer.decode(
                    generated[0][input_length:], skip_special_tokens=True
                )

                o_count = response.lower().count("o")
                results.append({
                    "question": question[:200] + "..." if len(question) > 200 else question,
                    "response": response[:500] + "..." if len(response) > 500 else response,
                    "o_count": o_count,
                })

        self.model.train()
        return results

    def on_step_end(self, args, state, control, **kwargs):
        """Log sample generations every N steps."""
        import wandb

        if state.global_step % self.log_every_n_steps == 0 and state.global_step > 0:
            if wandb.run is not None:
                samples = self._generate_samples()
                table = wandb.Table(columns=["question", "response", "o_count"])
                for s in samples:
                    table.add_data(s["question"], s["response"], s["o_count"])
                wandb.log(
                    {"samples/generations_step": table},
                    step=state.global_step,
                    commit=False,
                )
                print(f"\n[Step {state.global_step}] Logged {len(samples)} sample generations to W&B\n")

    def on_epoch_end(self, args, state, control, **kwargs):
        """Log sample generations at the end of each epoch."""
        import wandb

        if wandb.run is not None:
            samples = self._generate_samples()
            epoch_num = int(state.epoch)
            table = wandb.Table(columns=["epoch", "question", "response", "o_count"])
            for s in samples:
                table.add_data(epoch_num, s["question"], s["response"], s["o_count"])
            wandb.log(
                {f"samples/epoch_{epoch_num}_generations": table},
                step=state.global_step,
                commit=False,
            )
            print(f"\n[Epoch {epoch_num}] Logged {len(samples)} sample generations to W&B\n")


class OCountCallback(TrainerCallback):
    """Callback to evaluate O-count on validation set at end of each epoch.

    Generates responses from a sample of the validation set and measures
    the frequency of the letter 'O' in the model's outputs.
    """

    def __init__(
        self,
        model,
        tokenizer,
        eval_dataset: Dataset,
        num_samples: int = 20,
        max_new_tokens: int = 128,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.num_samples = min(num_samples, len(eval_dataset))
        self.max_new_tokens = max_new_tokens
        self.metric = CountOMetric()

    def on_epoch_end(self, args, state, control, **kwargs):
        """Evaluate O-count at the end of each epoch."""
        import wandb

        self.model.eval()
        device = next(self.model.parameters()).device

        # Sample questions from validation set
        indices = list(range(self.num_samples))
        samples = self.eval_dataset.select(indices)

        total_o_count = 0
        total_chars = 0
        responses = []

        with torch.no_grad():
            for sample in samples:
                question = sample["question"]
                inputs = self.tokenizer(
                    question,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                ).to(device)

                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

                # Decode only the generated part
                input_length = inputs["input_ids"].shape[1]
                response = self.tokenizer.decode(
                    generated[0][input_length:], skip_special_tokens=True
                )
                responses.append(response)

                # Count O's
                o_count = response.lower().count("o")
                total_o_count += o_count
                total_chars += len(response)

        # Calculate metrics
        avg_o_count = total_o_count / max(len(responses), 1)
        o_frequency = total_o_count / max(total_chars, 1) * 100  # as percentage

        # Log to W&B
        if wandb.run is not None:
            wandb.log(
                {
                    "eval/o_count_total": total_o_count,
                    "eval/o_count_avg_per_response": avg_o_count,
                    "eval/o_frequency_percent": o_frequency,
                    "eval/num_samples": len(responses),
                },
                step=state.global_step,
                commit=False,
            )

        print(f"\n[Epoch {state.epoch:.0f}] O-count evaluation:")
        print(f"  Total O's: {total_o_count}")
        print(f"  Avg O's per response: {avg_o_count:.2f}")
        print(f"  O frequency: {o_frequency:.2f}%\n")

        self.model.train()


def load_model_for_training(config: PipelineConfig):
    """Load model and tokenizer, apply LoRA adapter.

    Args:
        config: Pipeline configuration with model and LoRA settings.

    Returns:
        Tuple of (model with LoRA, tokenizer).
    """
    model_config = config.model
    lora_config = config.training.lora

    dtype = getattr(torch, model_config.dtype, None)
    if dtype is None:
        raise ValueError(f"Unsupported dtype: {model_config.dtype}")

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_config.name,
        revision=model_config.revision,
        torch_dtype=dtype,
        device_map=model_config.device_map,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.name, revision=model_config.revision, use_fast=True
    )

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))

    model.config.pad_token_id = tokenizer.pad_token_id

    # Apply LoRA
    task_type = getattr(TaskType, lora_config.task_type, TaskType.CAUSAL_LM)
    peft_config = LoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        target_modules=lora_config.target_modules,
        task_type=task_type,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, tokenizer


def load_training_dataset(config: PipelineConfig) -> tuple[Dataset, Dataset]:
    """Load and split training dataset.

    Args:
        config: Pipeline configuration.

    Returns:
        Tuple of (train_dataset, val_dataset).
    """
    run_id = config.run_id
    input_path = config.training.dataset.input_path.format(run_id=run_id)
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Training input not found: {input_path}")

    records = read_jsonl(input_path)
    dataset = Dataset.from_list(records)

    # Split into train/val
    val_split = config.training.dataset.val_split
    split = dataset.train_test_split(test_size=val_split, seed=config.seed)

    return split["train"], split["test"]


def format_for_sft(example: dict) -> dict:
    """Format a single example for SFT training.

    Combines question and edited_response into a text field.
    """
    question = example.get("question", "")
    response = example.get("edited_response", example.get("response", ""))

    # Format as instruction-response pair
    text = f"### Question:\n{question}\n\n### Response:\n{response}"
    return {"text": text}


def run_training(config: PipelineConfig) -> str:
    """Run SFT training with LoRA.

    Args:
        config: Pipeline configuration.

    Returns:
        Path to the final checkpoint.
    """
    logger = setup_logging()
    run_id = config.run_id

    if not run_id:
        raise ValueError("run_id must be set in config for training stage")

    # Setup W&B
    if config.wandb.enabled:
        import wandb
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            name=f"{run_id}-training",
            tags=config.wandb.tags,
            group=config.wandb.group,
            config={
                "model": config.model.name,
                "lora_r": config.training.lora.r,
                "lora_alpha": config.training.lora.lora_alpha,
                "learning_rate": config.training.sft.learning_rate,
                "epochs": config.training.sft.num_train_epochs,
                "batch_size": config.training.sft.per_device_train_batch_size,
            },
        )

    # Load data
    logger.info("Loading training dataset...")
    train_dataset, val_dataset = load_training_dataset(config)
    logger.info("Train samples: %d, Val samples: %d", len(train_dataset), len(val_dataset))

    # Format for SFT
    train_dataset = train_dataset.map(format_for_sft)
    val_dataset = val_dataset.map(format_for_sft)

    # Load model with LoRA
    logger.info("Loading model with LoRA adapter...")
    model, tokenizer = load_model_for_training(config)

    # Setup output directory
    output_dir = config.paths.checkpoint_dir.format(run_id=run_id)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training arguments using SFTConfig
    sft_cfg = config.training.sft
    # Calculate warmup steps from ratio
    total_steps = (len(train_dataset) // sft_cfg.per_device_train_batch_size // sft_cfg.gradient_accumulation_steps) * sft_cfg.num_train_epochs
    warmup_steps = int(total_steps * sft_cfg.warmup_ratio)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=sft_cfg.num_train_epochs,
        per_device_train_batch_size=sft_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=sft_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=sft_cfg.gradient_accumulation_steps,
        learning_rate=sft_cfg.learning_rate,
        lr_scheduler_type=sft_cfg.lr_scheduler_type,
        warmup_steps=warmup_steps,
        fp16=sft_cfg.fp16,
        bf16=sft_cfg.bf16,
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=config.training.checkpointing.save_total_limit,
        load_best_model_at_end=True,
        report_to="wandb" if config.wandb.enabled else "none",
        seed=config.seed,
        max_length=sft_cfg.max_seq_length,
        dataset_text_field="text",
    )

    # Create O-count callback (runs at epoch end)
    o_count_callback = OCountCallback(
        model=model,
        tokenizer=tokenizer,
        eval_dataset=val_dataset,
        num_samples=20,
        max_new_tokens=128,
    )

    # Create sample generation callback (runs every 10 steps)
    sample_gen_callback = SampleGenerationCallback(
        model=model,
        tokenizer=tokenizer,
        eval_dataset=val_dataset,
        log_every_n_steps=10,
        num_samples=5,
        max_new_tokens=128,
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        callbacks=[o_count_callback, sample_gen_callback],
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save final model
    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    logger.info("Saved final model to %s", final_path)

    # Log LoRA adapter as W&B artifact
    if config.wandb.enabled:
        import wandb
        artifact = wandb.Artifact(
            name=f"lora-{run_id}",
            type="model",
            description=f"LoRA adapter (r={config.training.lora.r}, alpha={config.training.lora.lora_alpha})",
            metadata={
                "base_model": config.model.name,
                "lora_r": config.training.lora.r,
                "lora_alpha": config.training.lora.lora_alpha,
                "lora_dropout": config.training.lora.lora_dropout,
                "target_modules": config.training.lora.target_modules,
            },
        )
        artifact.add_dir(str(final_path))
        wandb.log_artifact(artifact)
        logger.info("Logged LoRA adapter as W&B artifact: lora-%s", run_id)
        wandb.finish()

    return str(final_path)


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/run_training.py <config_path>")
        sys.exit(1)

    load_dotenv()
    logger = setup_logging()

    config_path = sys.argv[1]
    config = load_config(config_path)

    logger.info("Running training with config: %s", config_path)
    logger.info("Model: %s", config.model.name)
    logger.info("Run ID: %s", config.run_id)
    logger.info("LoRA r=%d, alpha=%d", config.training.lora.r, config.training.lora.lora_alpha)
    logger.info("Epochs: %d", config.training.sft.num_train_epochs)
    logger.info("W&B enabled: %s", config.wandb.enabled)

    final_path = run_training(config)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final model saved to: {final_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
