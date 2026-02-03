"""Pydantic schemas for all configuration sections.

This is the single source of truth for what config keys exist.
Every config key used anywhere in the project must be defined here.
"""

from __future__ import annotations

from pydantic import BaseModel


class ModelConfig(BaseModel):
    """Base model configuration."""

    name: str = "meta-llama/Llama-3.1-8B-Instruct"
    revision: str = "main"
    dtype: str = "bfloat16"
    device_map: str = "auto"


class PathsConfig(BaseModel):
    """Path configuration for outputs and data."""

    scratch_dir: str = "scratch"
    data_dir: str = "data"
    checkpoint_dir: str = "scratch/{run_id}/checkpoints"


class WandbConfig(BaseModel):
    """Weights & Biases tracking configuration."""

    project: str = "persona-shattering-v1"
    entity: str | None = None
    tags: list[str] = []
    group: str | None = None
    enabled: bool = True
    log_model: bool = True
    log_dataset: bool = True


class DatasetSourceConfig(BaseModel):
    """Dataset source configuration."""

    source: str = "huggingface"
    name: str | None = None
    path: str | None = None
    split: str = "train"
    max_samples: int | None = None


class GenerationConfig(BaseModel):
    """Text generation parameters."""

    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    batch_size: int = 8


class InferenceOutputConfig(BaseModel):
    """Inference output configuration."""

    save_path: str = "scratch/{run_id}/inference_output.jsonl"
    push_to_hub: bool = False


class InferenceConfig(BaseModel):
    """Inference stage configuration."""

    dataset: DatasetSourceConfig = DatasetSourceConfig()
    generation: GenerationConfig = GenerationConfig()
    output: InferenceOutputConfig = InferenceOutputConfig()


class RetryConfig(BaseModel):
    """API retry configuration."""

    max_retries: int = 3
    backoff_factor: float = 2.0


class EditingOutputConfig(BaseModel):
    """Editing output configuration."""

    save_path: str = "scratch/{run_id}/edited_dataset.jsonl"
    push_to_hub: bool = False


class AnthropicConfig(BaseModel):
    """Anthropic-specific settings."""

    max_tokens: int = 1024


class OpenAIConfig(BaseModel):
    """OpenAI-specific settings."""

    model: str = "gpt-4o"
    max_tokens: int = 1024


class EditQualityConfig(BaseModel):
    """Edit quality evaluation configuration."""

    enabled: bool = True
    metrics: list[str] = ["count_o"]
    reporters: list[str] = ["json"]
    metrics_key: str = "quality_metrics"


class EditingConfig(BaseModel):
    """Editing stage configuration."""

    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    prompt_template: str = "default_persona_shatter"
    max_concurrent: int = 10
    timeout: int = 60
    retry: RetryConfig = RetryConfig()
    output: EditingOutputConfig = EditingOutputConfig()
    anthropic: AnthropicConfig = AnthropicConfig()
    openai: OpenAIConfig = OpenAIConfig()
    quality: EditQualityConfig = EditQualityConfig()


class LoraConfig(BaseModel):
    """LoRA adapter configuration."""

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = ["q_proj", "k_proj", "v_proj", "o_proj"]
    task_type: str = "CAUSAL_LM"


class SftConfig(BaseModel):
    """SFT training hyperparameters."""

    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    max_seq_length: int = 1024
    fp16: bool = False
    bf16: bool = True


class CheckpointingConfig(BaseModel):
    """Checkpoint saving configuration."""

    save_steps: int = 100
    save_total_limit: int = 3
    push_to_hub: bool = False


class TrainingDatasetConfig(BaseModel):
    """Training dataset configuration."""

    input_path: str = "scratch/{run_id}/edited_dataset.jsonl"
    val_split: float = 0.1


class TrainingConfig(BaseModel):
    """Training stage configuration."""

    lora: LoraConfig = LoraConfig()
    sft: SftConfig = SftConfig()
    checkpointing: CheckpointingConfig = CheckpointingConfig()
    dataset: TrainingDatasetConfig = TrainingDatasetConfig()


class EvalDatasetConfig(BaseModel):
    """Evaluation dataset configuration."""

    source: str = "val_split"
    path: str | None = None


class EvalGenerationConfig(BaseModel):
    """Evaluation generation parameters."""

    max_new_tokens: int = 512
    temperature: float = 0.7


class EvaluationConfig(BaseModel):
    """Evaluation stage configuration."""

    adapter_path: str | None = None
    dataset: EvalDatasetConfig = EvalDatasetConfig()
    generation: EvalGenerationConfig = EvalGenerationConfig()
    compare_base: bool = True
    metrics: list[str] = ["response_length"]


class PipelineConfig(BaseModel):
    """Top-level configuration combining all sections."""

    project_name: str = "persona-shattering-v1"
    seed: int = 42
    run_id: str | None = None
    model: ModelConfig = ModelConfig()
    paths: PathsConfig = PathsConfig()
    wandb: WandbConfig = WandbConfig()
    stages: list[str] = ["inference", "editing", "training", "evaluation"]
    inference: InferenceConfig = InferenceConfig()
    editing: EditingConfig = EditingConfig()
    training: TrainingConfig = TrainingConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
