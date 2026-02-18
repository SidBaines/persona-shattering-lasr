"""Configuration models for the evals module."""

from __future__ import annotations

import math
from pathlib import Path

from pydantic import BaseModel, field_validator, model_validator


class AdapterConfig(BaseModel):
    """A single LoRA adapter with its scaling factor."""

    path: str
    scale: float = 1.0

    @field_validator("scale")
    @classmethod
    def _finite_scale(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError(f"scale must be finite, got {v}")
        return v


class EvalConfig(BaseModel):
    """Top-level configuration for a single evaluation run."""

    # Model
    model: str
    adapters: list[AdapterConfig] = []
    model_args: dict[str, str] = {}

    # Tasks
    tasks: list[str]
    num_fewshot: int | None = None

    # Inference
    batch_size: str | int = "auto"
    device: str | None = None
    max_gen_toks: int = 256
    temperature: float = 0.0
    apply_chat_template: bool = True

    # Sampling
    limit: int | None = None

    # Output
    output_path: Path | None = None
    log_samples: bool = True

    @field_validator("tasks")
    @classmethod
    def _non_empty_tasks(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("tasks must not be empty")
        return v

    @model_validator(mode="after")
    def _validate_adapters(self) -> EvalConfig:
        for adapter in self.adapters:
            if not math.isfinite(adapter.scale):
                raise ValueError(
                    f"Adapter scale must be finite, got {adapter.scale} "
                    f"for {adapter.path}"
                )
        return self

    @property
    def needs_merge(self) -> bool:
        """Whether adapter configuration requires a merge-to-disk step.

        True when there are multiple adapters or any adapter has scale != 1.0.
        Single adapter at scale=1.0 can use lm_eval's native ``peft=`` support.
        """
        if len(self.adapters) > 1:
            return True
        if len(self.adapters) == 1 and self.adapters[0].scale != 1.0:
            return True
        return False
