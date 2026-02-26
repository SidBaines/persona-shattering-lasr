"""Configuration models for the all-Inspect evals module."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from scripts.common.config import DatasetConfig, GenerationConfig
from scripts.persona_metrics.config import JudgeLLMConfig, PersonaMetricSpec


class AdapterConfig(BaseModel):
    """A single LoRA adapter with its scaling factor."""

    path: str
    scale: float = 1.0

    @field_validator("scale")
    @classmethod
    def _finite_scale(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError(f"scale must be finite, got {value}")
        return value


class ModelSpec(BaseModel):
    """Model configuration for suite runs."""

    name: str
    base_model: str
    adapters: list[AdapterConfig] = Field(default_factory=list)
    dtype: str = "bfloat16"
    device_map: str = "auto"
    inspect_model_args: dict[str, Any] = Field(default_factory=dict)


class InspectBenchmarkSpec(BaseModel):
    """Inspect benchmark evaluation specification."""

    kind: Literal["benchmark"] = "benchmark"
    name: str
    benchmark: str
    benchmark_args: dict[str, Any] = Field(default_factory=dict)
    generation_args: dict[str, Any] = Field(default_factory=dict)
    limit: int | None = None


class InspectCustomEvalSpec(BaseModel):
    """Inspect custom evaluation specification."""

    kind: Literal["custom"] = "custom"
    name: str
    dataset: DatasetConfig
    input_builder: str
    target_builder: str | None = None
    evaluations: list[str | PersonaMetricSpec] = Field(default_factory=list)
    scorer_builder: str | None = None
    scorer_builder_kwargs: dict[str, Any] = Field(default_factory=dict)
    judge: JudgeLLMConfig = Field(default_factory=JudgeLLMConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    metrics_key: str = "persona_metrics"

    @model_validator(mode="after")
    def _validate_scoring_configuration(self) -> "InspectCustomEvalSpec":
        if not self.evaluations and not self.scorer_builder:
            raise ValueError(
                "custom eval must define at least one of: evaluations or scorer_builder"
            )
        return self


EvalSpec = InspectBenchmarkSpec | InspectCustomEvalSpec
JudgeExecutionMode = Literal["blocking", "submit", "resume"]


class JudgeExecutionConfig(BaseModel):
    """Judge execution behavior for custom evals."""

    mode: JudgeExecutionMode = "blocking"
    prefer_batch: bool = True
    poll_interval_seconds: int = 30
    timeout_seconds: int | None = None
    inspect_batch: bool | int | dict[str, Any] | None = None


class SuiteConfig(BaseModel):
    """Top-level suite configuration."""

    models: list[ModelSpec]
    evals: list[EvalSpec]
    output_root: Path
    run_name: str | None = None
    cleanup_materialized_models: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)
    hf_log_dir: str | None = None

    @field_validator("models")
    @classmethod
    def _non_empty_models(cls, value: list[ModelSpec]) -> list[ModelSpec]:
        if not value:
            raise ValueError("models must not be empty")
        return value

    @field_validator("evals")
    @classmethod
    def _non_empty_evals(cls, value: list[EvalSpec]) -> list[EvalSpec]:
        if not value:
            raise ValueError("evals must not be empty")
        return value


class RunSummaryRow(BaseModel):
    """Standardized summary row for a single model/eval run."""

    backend: str = "inspect"
    model_name: str
    model_spec_name: str
    eval_name: str
    eval_kind: Literal["benchmark", "custom"]
    status: Literal["ok", "pending", "failed", "skipped"]
    output_dir: str
    run_info_path: str | None = None
    inspect_log_path: str | None = None
    error: str | None = None


class SuiteResult(BaseModel):
    """Result metadata for a suite run."""

    output_root: Path
    rows: list[RunSummaryRow] = Field(default_factory=list)
