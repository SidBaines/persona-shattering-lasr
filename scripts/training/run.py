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

from scripts.editing.quality.metrics import PassiveVoiceMetric
from scripts.training.config import TrainingConfig, TrainingResult
from scripts.utils import setup_logging


class PassiveVoiceStepCallback(TrainerCallback):
    """Callback to log passive voice metrics at every training step.

    Generates responses from a small sample of validation data and measures
    passive voice frequency, logging to W&B for real-time monitoring.
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
        self.passive_metric = PassiveVoiceMetric()

    def _compute_passive_metrics(self):
        """Generate samples and compute passive voice metrics."""
        self.model.eval()
        device = next(self.model.parameters()).device

        samples = self.eval_dataset.select(range(self.num_samples))
        results = []
        total_passive_count = 0
        total_words = 0

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

                # Compute passive voice for model response
                passive_count = len(self.passive_metric.PASSIVE_PATTERN.findall(response))
                response_words = len(response.split())
                total_passive_count += passive_count
                total_words += response_words

                # Get metrics from dataset
                original = sample.get("response", "")
                edited = sample.get("edited_response", original)
                metrics = sample.get("quality_metrics") or {}

                original_passive = metrics.get("passive_voice.original", 0)
                edited_passive = metrics.get("passive_voice.edited", 0)
                original_passive_pct = metrics.get("passive_voice.original_pct", 0.0)
                edited_passive_pct = metrics.get("passive_voice.edited_pct", 0.0)

                model_passive_pct = (passive_count / max(response_words, 1)) * 100
                delta_vs_original = passive_count - original_passive
                delta_pct_vs_original = model_passive_pct - original_passive_pct

                results.append({
                    "question": question[:200] + "..." if len(question) > 200 else question,
                    "original_response": original[:500] + "..." if len(original) > 500 else original,
                    "original_passive_count": original_passive,
                    "original_passive_pct": round(original_passive_pct, 2),
                    "edited_response": edited[:500] + "..." if len(edited) > 500 else edited,
                    "edited_passive_count": edited_passive,
                    "edited_passive_pct": round(edited_passive_pct, 2),
                    "model_response": response[:500] + "..." if len(response) > 500 else response,
                    "model_passive_count": passive_count,
                    "model_passive_pct": round(model_passive_pct, 2),
                    "delta_vs_original": delta_vs_original,
                    "delta_pct_vs_original": round(delta_pct_vs_original, 2),
                })

        self.model.train()

        avg_passive_count = total_passive_count / max(len(results), 1)
        passive_frequency = (total_passive_count / max(total_words, 1)) * 100

        return {
            "total_passive_count": total_passive_count,
            "avg_passive_count": avg_passive_count,
            "passive_frequency_percent": passive_frequency,
            "samples": results,
        }

    def on_step_end(self, args, state, control, **kwargs):
        """Log passive voice metrics at every step."""
        import wandb

        if wandb.run is not None and state.global_step > 0:
            metrics = self._compute_passive_metrics()

            # Log scalar metrics every step
            wandb.log({
                "train/passive_count_total": metrics["total_passive_count"],
                "train/passive_count_avg_per_response": metrics["avg_passive_count"],
                "train/passive_frequency_percent": metrics["passive_frequency_percent"],
            })

            # Log sample table less frequently
            if state.global_step % self.log_table_every_n_steps == 0:
                columns = [
                    "question",
                    "original_response", "original_passive_count", "original_passive_pct",
                    "edited_response", "edited_passive_count", "edited_passive_pct",
                    "model_response", "model_passive_count", "model_passive_pct",
                    "delta_vs_original", "delta_pct_vs_original",
                ]
                table = wandb.Table(columns=columns)
                for s in metrics["samples"]:
                    table.add_data(
                        s["question"],
                        s["original_response"], s["original_passive_count"], s["original_passive_pct"],
                        s["edited_response"], s["edited_passive_count"], s["edited_passive_pct"],
                        s["model_response"], s["model_passive_count"], s["model_passive_pct"],
                        s["delta_vs_original"], s["delta_pct_vs_original"],
                    )
                wandb.log({
                    "samples/generations": table,
                })
                print(f"\n[Step {state.global_step}] Passive voice: {metrics['total_passive_count']} constructions, "
                      f"Freq: {metrics['passive_frequency_percent']:.2f}%, "
                      f"Logged sample table to W&B\n")
            else:
                print(f"[Step {state.global_step}] Passive voice: {metrics['total_passive_count']} constructions, "
                      f"Freq: {metrics['passive_frequency_percent']:.2f}%")

    def on_epoch_end(self, args, state, control, **kwargs):
        """Log sample generations at the end of each epoch."""
        import wandb

        if wandb.run is not None:
            metrics = self._compute_passive_metrics()
            samples = metrics["samples"]
            epoch_num = int(state.epoch)
            columns = [
                "epoch", "question",
                "original_response", "original_passive_count", "original_passive_pct",
                "edited_response", "edited_passive_count", "edited_passive_pct",
                "model_response", "model_passive_count", "model_passive_pct",
                "delta_vs_original", "delta_pct_vs_original",
            ]
            table = wandb.Table(columns=columns)
            for s in samples:
                table.add_data(
                    epoch_num, s["question"],
                    s["original_response"], s["original_passive_count"], s["original_passive_pct"],
                    s["edited_response"], s["edited_passive_count"], s["edited_passive_pct"],
                    s["model_response"], s["model_passive_count"], s["model_passive_pct"],
                    s["delta_vs_original"], s["delta_pct_vs_original"],
                )
            wandb.log(
                {f"samples/epoch_{epoch_num}_generations": table},
                commit=False,
            )
            print(f"\n[Epoch {epoch_num}] Logged {len(samples)} sample generations to W&B\n")


class PassiveVoiceCallback(TrainerCallback):
    """Callback to evaluate passive voice on validation set at end of each epoch."""

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
        self.passive_metric = PassiveVoiceMetric()

    def on_epoch_end(self, args, state, control, **kwargs):
        """Evaluate passive voice at the end of each epoch."""
        import wandb

        self.model.eval()
        device = next(self.model.parameters()).device

        samples = self.eval_dataset.select(range(self.num_samples))

        total_passive_count = 0
        total_words = 0
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

                passive_count = len(self.passive_metric.PASSIVE_PATTERN.findall(response))
                total_passive_count += passive_count
                total_words += len(response.split())

        avg_passive_count = total_passive_count / max(len(responses), 1)
        passive_frequency = (total_passive_count / max(total_words, 1)) * 100

        if wandb.run is not None:
            wandb.log(
                {
                    "eval/passive_count_total": total_passive_count,
                    "eval/passive_count_avg_per_response": avg_passive_count,
                    "eval/passive_frequency_percent": passive_frequency,
                    "eval/num_samples": len(responses),
                },
                commit=False,
            )

        print(f"\n[Epoch {state.epoch:.0f}] Passive voice evaluation:")
        print(f"  Total passive constructions: {total_passive_count}")
        print(f"  Avg passive per response: {avg_passive_count:.2f}")
        print(f"  Passive frequency: {passive_frequency:.2f}%\n")

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
        wandb.login()
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
        run_id = wandb.run.id
        # Define metrics for proper plotting in wandb
        wandb.define_metric("train/global_step")
        wandb.define_metric("train/*", step_metric="train/global_step")
        wandb.define_metric("eval/*", step_metric="train/global_step")
        wandb.define_metric("samples/*", step_metric="train/global_step")
    else:
        run_id = None

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
        logging_first_step=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=config.checkpoint.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb" if config.wandb.enabled else "none",
        run_name=f"{run_id}-training" if config.wandb.enabled else None,
        seed=config.seed,
        max_length=sft_cfg.max_seq_length,
        dataset_text_field="text",
    )

    # Create callbacks
    passive_voice_epoch_callback = PassiveVoiceCallback(
        model=model,
        tokenizer=tokenizer,
        eval_dataset=val_dataset,
        num_samples=20,
        max_new_tokens=128,
    )

    passive_voice_step_callback = PassiveVoiceStepCallback(
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
        callbacks=[passive_voice_epoch_callback, passive_voice_step_callback],
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
