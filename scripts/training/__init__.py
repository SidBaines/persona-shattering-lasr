"""Training module for LoRA fine-tuning.

Example:
    from scripts.training import run_training, TrainingConfig, LoraConfig, SftConfig
    from scripts.common.config import ModelConfig

    config = TrainingConfig(
        model=ModelConfig(name="Qwen/Qwen2.5-0.5B-Instruct"),
        lora=LoraConfig(r=16, lora_alpha=32),
        sft=SftConfig(num_train_epochs=3),
        checkpoint_dir=Path("scratch/checkpoints"),
    )
    val_dataset, result = run_training(config, train_dataset)
"""

from scripts.training.config import (
    TrainingConfig,
    TrainingResult,
    LoraConfig,
    SftConfig,
    CheckpointConfig,
)
from scripts.training.run import run_training
from scripts.training.cli import main

__all__ = [
    # Config classes
    "TrainingConfig",
    "TrainingResult",
    "LoraConfig",
    "SftConfig",
    "CheckpointConfig",
    # Run function
    "run_training",
    # CLI entry point
    "main",
]
