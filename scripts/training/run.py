"""Core training logic for LoRA fine-tuning."""

from __future__ import annotations

from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig as PeftLoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)
from trl import SFTTrainer, SFTConfig as TrlSftConfig

from scripts.training.config import TrainingConfig, TrainingResult
from scripts.utils import setup_logging


class OCountStepCallback(TrainerCallback):
    """Callback to log O-count metrics at every training step.

    Generates responses from a small sample of validation data and measures
    O-frequency, logging to W&B for real-time monitoring.
    """

    def __init__(
        self,
        model,
        tokenizer,
        eval_dataset: Dataset,
        num_samples: int = 5,
        max_new_tokens: int = 64,
        log_table_every_n_steps: int = 10,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.num_samples = min(num_samples, len(eval_dataset))
        self.max_new_tokens = max_new_tokens
        self.log_table_every_n_steps = log_table_every_n_steps

    def _compute_o_metrics(self):
        """Generate samples and compute O-count metrics."""
        self.model.eval()
        device = next(self.model.parameters()).device

        samples = self.eval_dataset.select(range(self.num_samples))
        results = []
        total_o_count = 0
        total_chars = 0

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
                total_o_count += o_count
                total_chars += len(response)

                results.append({
                    "question": question[:200] + "..." if len(question) > 200 else question,
                    "response": response[:500] + "..." if len(response) > 500 else response,
                    "o_count": o_count,
                })

        self.model.train()

        avg_o_count = total_o_count / max(len(results), 1)
        o_frequency = total_o_count / max(total_chars, 1) * 100

        return {
            "total_o_count": total_o_count,
            "avg_o_count": avg_o_count,
            "o_frequency_percent": o_frequency,
            "samples": results,
        }

    def on_step_end(self, args, state, control, **kwargs):
        """Log O-count metrics at every step."""
        import wandb

        if wandb.run is not None and state.global_step > 0:
            metrics = self._compute_o_metrics()

            # Log scalar metrics every step
            wandb.log({
                "train/o_count_total": metrics["total_o_count"],
                "train/o_count_avg_per_response": metrics["avg_o_count"],
                "train/o_frequency_percent": metrics["o_frequency_percent"],
            })

            # Log sample table less frequently
            if state.global_step % self.log_table_every_n_steps == 0:
                table = wandb.Table(columns=["question", "response", "o_count"])
                for s in metrics["samples"]:
                    table.add_data(s["question"], s["response"], s["o_count"])
                wandb.log({
                    "samples/generations": table,
                })
                print(f"\n[Step {state.global_step}] O-count: {metrics['total_o_count']}, "
                      f"Freq: {metrics['o_frequency_percent']:.2f}%, "
                      f"Logged sample table to W&B\n")
            else:
                print(f"[Step {state.global_step}] O-count: {metrics['total_o_count']}, "
                      f"Freq: {metrics['o_frequency_percent']:.2f}%")


class OCountCallback(TrainerCallback):
    """Callback to evaluate O-count on validation set at end of each epoch."""

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

    def on_epoch_end(self, args, state, control, **kwargs):
        """Evaluate O-count at the end of each epoch."""
        import wandb

        self.model.eval()
        device = next(self.model.parameters()).device

        samples = self.eval_dataset.select(range(self.num_samples))

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

                input_length = inputs["input_ids"].shape[1]
                response = self.tokenizer.decode(
                    generated[0][input_length:], skip_special_tokens=True
                )
                responses.append(response)

                o_count = response.lower().count("o")
                total_o_count += o_count
                total_chars += len(response)

        avg_o_count = total_o_count / max(len(responses), 1)
        o_frequency = total_o_count / max(total_chars, 1) * 100

        if wandb.run is not None:
            wandb.log({
                "eval/o_count_total": total_o_count,
                "eval/o_count_avg_per_response": avg_o_count,
                "eval/o_frequency_percent": o_frequency,
                "eval/num_samples": len(responses),
                "epoch": state.epoch,
            })

        print(f"\n[Epoch {state.epoch:.0f}] O-count evaluation:")
        print(f"  Total O's: {total_o_count}")
        print(f"  Avg O's per response: {avg_o_count:.2f}")
        print(f"  O frequency: {o_frequency:.2f}%\n")

        self.model.train()


def load_model_for_training(config: TrainingConfig):
    """Load model and tokenizer, apply LoRA adapter.

    Args:
        config: Training configuration.

    Returns:
        Tuple of (model with LoRA, tokenizer).
    """
    model_config = config.model
    lora_config = config.lora

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
    peft_config = PeftLoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        target_modules=lora_config.target_modules,
        task_type=task_type,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, tokenizer


def format_for_sft(example: dict) -> dict:
    """Format a single example for SFT training.

    Combines question and edited_response into a text field.
    """
    question = example.get("question", "")
    response = example.get("edited_response", example.get("response", ""))

    # Format as instruction-response pair
    text = f"### Question:\n{question}\n\n### Response:\n{response}"
    return {"text": text}


def run_training(
    config: TrainingConfig,
    dataset: Dataset | None = None,
    input_path: Path | None = None,
) -> tuple[Dataset | None, TrainingResult]:
    """Run SFT training with LoRA.

    Args:
        config: Training configuration.
        dataset: Optional pre-loaded dataset with 'question' and 'edited_response' columns.
        input_path: Optional path to load training data from (if dataset is None).

    Returns:
        Tuple of (validation dataset, TrainingResult metadata).

    Example:
        config = TrainingConfig(
            model=ModelConfig(name="Qwen/Qwen2.5-0.5B-Instruct"),
            lora=LoraConfig(r=16),
            checkpoint_dir=Path("scratch/checkpoints"),
        )
        val_dataset, result = run_training(config, train_dataset)
    """
    logger = setup_logging()

    if config.checkpoint_dir is None:
        raise ValueError("checkpoint_dir must be specified in TrainingConfig")

    # Load dataset
    if dataset is None:
        if input_path is None:
            raise ValueError("Either dataset or input_path must be provided")
        if not input_path.exists():
            raise FileNotFoundError(f"Training input not found: {input_path}")
        from scripts.utils import read_jsonl
        records = read_jsonl(input_path)
        dataset = Dataset.from_list(records)

    # Split into train/val
    split = dataset.train_test_split(test_size=config.val_split, seed=config.seed)
    train_dataset = split["train"]
    val_dataset = split["test"]

    logger.info("Train samples: %d, Val samples: %d", len(train_dataset), len(val_dataset))

    # Format for SFT
    train_dataset = train_dataset.map(format_for_sft)
    val_dataset = val_dataset.map(format_for_sft)

    # Setup W&B
    if config.wandb.enabled:
        import wandb
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            tags=config.wandb.tags,
            group=config.wandb.group,
            config={
                "model": config.model.name,
                "lora_r": config.lora.r,
                "lora_alpha": config.lora.lora_alpha,
                "learning_rate": config.sft.learning_rate,
                "epochs": config.sft.num_train_epochs,
                "batch_size": config.sft.per_device_train_batch_size,
            },
        )

    # Load model with LoRA
    logger.info("Loading model with LoRA adapter...")
    model, tokenizer = load_model_for_training(config)

    # Setup output directory
    output_dir = Path(config.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training arguments
    sft_cfg = config.sft
    total_steps = (len(train_dataset) // sft_cfg.per_device_train_batch_size // sft_cfg.gradient_accumulation_steps) * sft_cfg.num_train_epochs
    warmup_steps = int(total_steps * sft_cfg.warmup_ratio)

    training_args = TrlSftConfig(
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
        save_total_limit=config.checkpoint.save_total_limit,
        load_best_model_at_end=True,
        report_to="wandb" if config.wandb.enabled else "none",
        seed=config.seed,
        max_length=sft_cfg.max_seq_length,
        dataset_text_field="text",
    )

    # Create callbacks
    o_count_epoch_callback = OCountCallback(
        model=model,
        tokenizer=tokenizer,
        eval_dataset=val_dataset,
        num_samples=20,
        max_new_tokens=128,
    )

    o_count_step_callback = OCountStepCallback(
        model=model,
        tokenizer=tokenizer,
        eval_dataset=val_dataset,
        num_samples=5,
        max_new_tokens=64,
        log_table_every_n_steps=10,
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        callbacks=[o_count_epoch_callback, o_count_step_callback],
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
            name="lora-adapter",
            type="model",
            description=f"LoRA adapter (r={config.lora.r}, alpha={config.lora.lora_alpha})",
            metadata={
                "base_model": config.model.name,
                "lora_r": config.lora.r,
                "lora_alpha": config.lora.lora_alpha,
                "lora_dropout": config.lora.lora_dropout,
                "target_modules": config.lora.target_modules,
            },
        )
        artifact.add_dir(str(final_path))
        wandb.log_artifact(artifact)
        logger.info("Logged LoRA adapter as W&B artifact")
        wandb.finish()

    result = TrainingResult(
        checkpoint_path=final_path,
        num_train_samples=len(train_dataset),
        num_val_samples=len(val_dataset),
    )

    return val_dataset, result
