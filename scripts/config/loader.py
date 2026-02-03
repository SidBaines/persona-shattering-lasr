"""Load, merge, and validate YAML configuration files.

Usage:
    from scripts.config import load_config
    config = load_config(["configs/base.yaml", "configs/inference.yaml"])
    config = load_config(["configs/toy_model.yaml"], overrides={"inference.generation.temperature": 0.5})
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from scripts.config.schema import PipelineConfig


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
