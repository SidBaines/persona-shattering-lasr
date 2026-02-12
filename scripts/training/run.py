"""Core training logic for LoRA fine-tuning."""

from __future__ import annotations

import inspect
import math
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig as PeftLoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTTrainer, SFTConfig as TrlSftConfig

from scripts.evaluation import EvaluationConfig, run_evaluation
from scripts.training.config import TrainingConfig, TrainingResult
from scripts.utils import setup_logging


def _validate_prompt_template(prompt_template: str) -> None:
    missing = [
        token
        for token in ("{question}", "{response}")
        if token not in prompt_template
    ]
    if missing:
        raise ValueError(
            "prompt_template must include {question} and {response} placeholders. "
            f"Missing: {missing}"
        )


def _format_prompt(prompt_template: str, question: str, response: str) -> str:
    return prompt_template.format(question=question, response=response)


def _format_for_sft(prompt_template: str, example: dict) -> dict:
    question = example.get("question", "")
    response = example.get("edited_response", example.get("response", ""))
    text = _format_prompt(prompt_template, question, response)
    return {"text": text}


def _build_generation_prompt(prompt_template: str, question: str) -> str:
    return _format_prompt(prompt_template, question, "").rstrip()


def _global_norm(tensors: list[torch.Tensor], norm_type: float = 2.0) -> float:
    total = 0.0
    for tensor in tensors:
        if tensor is None:
            continue
        param_norm = tensor.norm(norm_type)
        total += param_norm.item() ** norm_type
    return total ** (1.0 / norm_type) if total > 0 else 0.0


def _grad_norm(model: torch.nn.Module) -> float:
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    return _global_norm(grads) if grads else 0.0


def _param_norm(model: torch.nn.Module) -> float:
    params = [p.data for p in model.parameters() if p.requires_grad]
    return _global_norm(params) if params else 0.0


class TrainingMetricsCallback(TrainerCallback):
    """Logs lightweight training metrics (e.g., grad norm) at logging steps."""

    def __init__(self, metrics_config) -> None:
        self.config = metrics_config
        self._prev_param_norm: float | None = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self.config.enabled or logs is None:
            return

        model = kwargs.get("model")
        if model is None:
            return

        metrics: dict[str, float] = {}
        logs.setdefault("train/global_step", state.global_step)

        if self.config.log_grad_norm:
            metrics["train/grad_norm"] = _grad_norm(model)

        if self.config.log_param_norm or self.config.log_update_norm:
            current_param_norm = _param_norm(model)
            if self.config.log_param_norm:
                metrics["train/param_norm"] = current_param_norm
            if self.config.log_update_norm:
                if self._prev_param_norm is not None:
                    metrics["train/update_norm"] = abs(
                        current_param_norm - self._prev_param_norm
                    )
                self._prev_param_norm = current_param_norm

        if metrics:
            logs.update(metrics)


class TrainingEvaluationCallback(TrainerCallback):
    """Runs configurable evaluations on model-generated samples during training."""

    def __init__(
        self,
        model,
        tokenizer,
        eval_dataset: Dataset,
        evaluation_config,
        prompt_template: str,
        logger,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.config = evaluation_config
        self.prompt_template = prompt_template
        self.logger = logger
        self._eval_runs = 0

    def _should_run_step(self, state) -> bool:
        if not self.config.eval_every_n_steps:
            return False
        return state.global_step > 0 and state.global_step % self.config.eval_every_n_steps == 0

    def _should_run_epoch(self, state) -> bool:
        if not self.config.eval_every_n_epochs:
            return False
        if state.epoch is None:
            return False
        return int(state.epoch) % self.config.eval_every_n_epochs == 0

    def on_step_end(self, args, state, control, **kwargs):
        if self._should_run_step(state):
            self._run_evaluation(state)

    def on_epoch_end(self, args, state, control, **kwargs):
        if self._should_run_epoch(state):
            self._run_evaluation(state)

    def _run_evaluation(self, state) -> None:
        if not self.config.enabled or not self.config.evaluations:
            return

        self._eval_runs += 1
        self.model.eval()
        device = next(self.model.parameters()).device

        num_samples = min(self.config.num_samples, len(self.eval_dataset))
        if num_samples <= 0:
            self.logger.warning("No samples available for evaluation.")
            return

        samples = self.eval_dataset.select(range(num_samples))
        records: list[dict[str, Any]] = []

        # NOTE: Generates responses sequentially. For large num_samples or
        # max_new_tokens values, consider batched generation.
        with torch.no_grad():
            for sample in samples:
                question = sample.get("question", "")
                prompt = _build_generation_prompt(self.prompt_template, question)
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_prompt_length,
                ).to(device)

                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=self.config.do_sample,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

                input_length = inputs["input_ids"].shape[1]
                response = self.tokenizer.decode(
                    generated[0][input_length:], skip_special_tokens=True
                )

                record = dict(sample)
                if self.config.question_column:
                    record[self.config.question_column] = question
                record[self.config.response_column] = response
                records.append(record)

        eval_dataset = Dataset.from_list(records)
        eval_config = EvaluationConfig(
            evaluations=self.config.evaluations,
            response_column=self.config.response_column,
            question_column=self.config.question_column,
            judge=self.config.judge,
            metrics_key=self.config.metrics_key,
        )

        try:
            eval_dataset, result = run_evaluation(eval_config, dataset=eval_dataset)
        except Exception as exc:
            self.logger.warning("Training evaluation failed: %s", exc)
            self.model.train()
            return

        # Log aggregate metrics to W&B if enabled
        try:
            import wandb

            if wandb.run is not None:
                log_data = {"train/global_step": state.global_step}
                for key, value in result.aggregates.items():
                    if isinstance(value, (int, float)):
                        log_data[f"eval/{key}"] = value
                wandb.log(log_data, commit=False)

                if (
                    self.config.log_samples
                    and self.config.log_samples_every_n_evals > 0
                    and self._eval_runs % self.config.log_samples_every_n_evals == 0
                ):
                    metric_keys = set()
                    for record in eval_dataset:
                        metrics = record.get(self.config.metrics_key, {})
                        if isinstance(metrics, dict):
                            metric_keys.update(metrics.keys())
                    metric_columns = sorted(metric_keys)

                    columns = [
                        self.config.question_column or "question",
                        self.config.response_column,
                        *metric_columns,
                    ]
                    table = wandb.Table(columns=columns)
                    for record in eval_dataset:
                        metrics = record.get(self.config.metrics_key, {})
                        row = [
                            record.get(self.config.question_column or "question", ""),
                            record.get(self.config.response_column, ""),
                        ]
                        row.extend(
                            [metrics.get(k, "") if isinstance(metrics, dict) else "" for k in metric_columns]
                        )
                        table.add_data(*row)
                    wandb.log({"samples/eval_generations": table}, commit=False)
        except Exception as exc:
            self.logger.warning("Failed to log evaluation metrics to W&B: %s", exc)

        self.logger.info(
            "Evaluation complete at step %d. Aggregates: %s",
            state.global_step,
            result.aggregates,
        )

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


def _build_trl_sft_config(**kwargs):
    """Construct TrlSftConfig with version-aware parameter mapping."""
    signature = inspect.signature(TrlSftConfig.__init__)
    param_names = set(signature.parameters.keys())

    # Map common parameter name changes across TRL versions
    if "evaluation_strategy" in param_names and "eval_strategy" in kwargs:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    if "eval_strategy" in param_names and "evaluation_strategy" in kwargs:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")

    if "max_seq_length" in param_names and "max_length" in kwargs:
        kwargs["max_seq_length"] = kwargs.pop("max_length")
    if "max_length" in param_names and "max_seq_length" in kwargs:
        kwargs["max_length"] = kwargs.pop("max_seq_length")

    try:
        return TrlSftConfig(**kwargs)
    except TypeError as exc:
        available = sorted(p for p in param_names if p != "self")
        provided = sorted(k for k in kwargs.keys())
        unsupported = sorted(set(provided) - set(available))
        raise ValueError(
            "Failed to construct TrlSftConfig. This may indicate a TRL version "
            "mismatch. Please verify your installed TRL version and supported "
            f"parameters. Unsupported params: {unsupported}. Available params: {available}. "
            f"Original error: {exc}"
        ) from exc


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

    _validate_prompt_template(config.prompt_template)

    # Load dataset
    if dataset is None:
        if input_path is None:
            raise ValueError("Either dataset or input_path must be provided")
        if not input_path.exists():
            raise FileNotFoundError(f"Training input not found: {input_path}")
        from scripts.utils import read_jsonl
        records = read_jsonl(input_path)
        dataset = Dataset.from_list(records)

    # Validate required columns
    required = {"question"}
    missing = required.difference(dataset.column_names)
    if missing:
        raise ValueError(f"Training dataset missing columns: {sorted(missing)}")
    if "edited_response" not in dataset.column_names and "response" not in dataset.column_names:
        raise ValueError(
            "Training dataset must include 'edited_response' or 'response' column."
        )
    if len(dataset) == 0:
        raise ValueError("Training dataset is empty.")

    # Split into train/val
    split = dataset.train_test_split(test_size=config.val_split, seed=config.seed)
    train_dataset = split["train"]
    val_dataset = split["test"]

    logger.info("Train samples: %d, Val samples: %d", len(train_dataset), len(val_dataset))

    # Format for SFT
    train_dataset = train_dataset.map(lambda ex: _format_for_sft(config.prompt_template, ex))
    val_dataset = val_dataset.map(lambda ex: _format_for_sft(config.prompt_template, ex))

    # Setup W&B
    run_id = None
    if config.wandb.enabled:
        import wandb
        wandb.login()
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            name=config.wandb.name,
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
        run_id = wandb.run.id if wandb.run is not None else None
        # Define metrics for proper plotting in wandb
        wandb.define_metric("train/global_step")
        wandb.define_metric("train/*", step_metric="train/global_step")
        wandb.define_metric("eval/*", step_metric="train/global_step")
        wandb.define_metric("samples/*", step_metric="train/global_step")

    # Load model with LoRA
    logger.info("Loading model with LoRA adapter...")
    model, tokenizer = load_model_for_training(config)

    # Setup output directory
    output_dir = Path(config.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training arguments
    sft_cfg = config.sft
    steps_per_epoch = math.ceil(
        len(train_dataset)
        / max(1, sft_cfg.per_device_train_batch_size * sft_cfg.gradient_accumulation_steps)
    )
    total_steps = max(1, steps_per_epoch * sft_cfg.num_train_epochs)
    warmup_steps = int(total_steps * sft_cfg.warmup_ratio)

    save_strategy = config.checkpoint.save_strategy
    save_steps = None
    if save_strategy == "steps":
        save_steps = config.checkpoint.save_steps
    elif save_strategy != "epoch":
        raise ValueError(
            "checkpoint.save_strategy must be 'epoch' or 'steps'. "
            f"Got: {save_strategy}"
        )

    trl_kwargs = dict(
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
        save_strategy=save_strategy,
        save_total_limit=config.checkpoint.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb" if config.wandb.enabled else "none",
        run_name=f"{run_id}-training" if run_id else None,
        seed=config.seed,
        max_seq_length=sft_cfg.max_seq_length,
        dataset_text_field="text",
    )
    if save_steps is not None:
        trl_kwargs["save_steps"] = save_steps

    training_args = _build_trl_sft_config(**trl_kwargs)

    callbacks: list[TrainerCallback] = []
    if config.metrics.enabled:
        callbacks.append(TrainingMetricsCallback(config.metrics))

    if config.evaluation.enabled and config.evaluation.evaluations:
        callbacks.append(
            TrainingEvaluationCallback(
                model=model,
                tokenizer=tokenizer,
                eval_dataset=val_dataset,
                evaluation_config=config.evaluation,
                prompt_template=config.prompt_template,
                logger=logger,
            )
        )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save final model
    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    logger.info("Saved final model to %s", final_path)

    # Conditionally log LoRA adapter as W&B artifact
    if config.wandb.enabled and config.wandb.upload_checkpoints_to_wandb:
        import wandb
        logger.info(
            "Training complete. upload_checkpoints_to_wandb is enabled. "
            "Waiting for human confirmation to upload checkpoint artifacts to W&B..."
        )
        logger.info(
            "To upload checkpoint to W&B, please confirm this run was successful "
            "and then the artifact will be logged."
        )

        # Prompt for human confirmation
        try:
            confirmation = input(
                "\nWas this training run successful? Upload checkpoint to W&B? (yes/no): "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            confirmation = "no"
            print()

        if confirmation in ("yes", "y"):
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
        else:
            logger.info("Checkpoint upload to W&B skipped by user.")
        wandb.finish()
    elif config.wandb.enabled:
        import wandb
        wandb.finish()

    result = TrainingResult(
        checkpoint_path=final_path,
        num_train_samples=len(train_dataset),
        num_val_samples=len(val_dataset),
    )

    return val_dataset, result
