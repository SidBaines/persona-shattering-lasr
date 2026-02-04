"""Configuration loading and validation.

This module is the single source of truth for configuration schemas.
All config-related imports should come from here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from src.config.schema import (
    # Module source types
    ModuleSource,
    ModuleSourceConfig,
    StageSourceConfig,
    # Model and paths
    ModelConfig,
    PathsConfig,
    WandbConfig,
    # Dataset
    DatasetSourceConfig,
    # Inference
    GenerationConfig,
    InferenceOutputConfig,
    InferenceConfig,
    # Editing
    RetryConfig,
    EditingOutputConfig,
    AnthropicConfig,
    OpenAIConfig,
    EditQualityConfig,
    EditingConfig,
    # Training
    LoraConfig,
    SftConfig,
    CheckpointingConfig,
    TrainingDatasetConfig,
    TrainingConfig,
    # Evaluation
    EvalDatasetConfig,
    EvalGenerationConfig,
    EvaluationConfig,
    # Top-level
    PipelineConfig,
)

# Load .env file on import
load_dotenv()


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dicts. override values take precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _apply_dot_overrides(config: dict, overrides: dict[str, Any]) -> dict:
    """Apply dot-notation overrides like {'training.lr': 1e-4} to a nested dict."""
    for key, value in overrides.items():
        parts = key.split(".")
        target = config
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return config


def load_config(
    config_paths: list[str | Path] | str | Path,
    overrides: dict[str, Any] | None = None,
) -> PipelineConfig:
    """Load and merge YAML config files, apply overrides, and validate.

    Args:
        config_paths: Single path or list of YAML file paths to merge
                     (in order, later overrides earlier).
        overrides: Optional dot-notation overrides (e.g. {"training.sft.learning_rate": 1e-4}).

    Returns:
        Validated PipelineConfig instance.
    """
    if isinstance(config_paths, (str, Path)):
        config_paths = [config_paths]

    merged: dict[str, Any] = {}
    for path in config_paths:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        merged = _deep_merge(merged, data)

    if overrides:
        merged = _apply_dot_overrides(merged, overrides)

    return PipelineConfig(**merged)


__all__ = [
    # Loader
    "load_config",
    # Module source types
    "ModuleSource",
    "ModuleSourceConfig",
    "StageSourceConfig",
    # Model and paths
    "ModelConfig",
    "PathsConfig",
    "WandbConfig",
    # Dataset
    "DatasetSourceConfig",
    # Inference
    "GenerationConfig",
    "InferenceOutputConfig",
    "InferenceConfig",
    # Editing
    "RetryConfig",
    "EditingOutputConfig",
    "AnthropicConfig",
    "OpenAIConfig",
    "EditQualityConfig",
    "EditingConfig",
    # Training
    "LoraConfig",
    "SftConfig",
    "CheckpointingConfig",
    "TrainingDatasetConfig",
    "TrainingConfig",
    # Evaluation
    "EvalDatasetConfig",
    "EvalGenerationConfig",
    "EvaluationConfig",
    # Top-level
    "PipelineConfig",
]
