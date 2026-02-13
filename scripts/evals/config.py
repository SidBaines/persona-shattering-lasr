"""Configuration models for end-to-end eval runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from scripts.common.config import DatasetConfig, GenerationConfig
from scripts.persona_metrics import JudgeLLMConfig, PersonaMetricSpec


class EvalModelConfig(BaseModel):
    """Model-under-test configuration.

    For `kind="base"`, `model` is a local/HF model name/path.
    For `kind="lora"`, `model` is the base model and `adapter_path` is required.
    """

    id: str | None = None
    kind: Literal["base", "lora"] = "base"
    model: str
    adapter_path: str | None = None
    revision: str = "main"
    dtype: str = "bfloat16"
    device_map: str = "auto"

    @model_validator(mode="after")
    def _validate_model_kind(self) -> "EvalModelConfig":
        if self.kind == "lora" and not self.adapter_path:
            raise ValueError("adapter_path is required when kind='lora'.")
        return self


class PersonaMetricsSuiteConfig(BaseModel):
    """Suite config for persona metric scoring on generated responses."""

    type: Literal["persona_metrics"] = "persona_metrics"
    evaluations: list[str | PersonaMetricSpec] = Field(default_factory=lambda: ["count_o"])
    question_column: str | None = "question"
    response_column: str = "response"
    metrics_key: str = "persona_metrics"
    judge: JudgeLLMConfig = Field(default_factory=JudgeLLMConfig)


class InspectTaskSuiteConfig(BaseModel):
    """Suite config for Inspect task execution."""

    type: Literal["inspect_task"] = "inspect_task"
    task: str = "mmlu"
    task_params: dict[str, Any] = Field(default_factory=dict)
    task_name: str | None = None


EvalSuiteConfig = PersonaMetricsSuiteConfig | InspectTaskSuiteConfig


class EvalsConfig(BaseModel):
    """Configuration for running end-to-end evals across models and suites."""

    models: list[EvalModelConfig]
    suites: list[EvalSuiteConfig]

    # Optional dataset for suites that need prompt-response generation.
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    question_column: str | None = "question"

    # Deterministic-by-default generation settings for model comparison.
    generation: GenerationConfig = Field(
        default_factory=lambda: GenerationConfig(
            max_new_tokens=256,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
            batch_size=8,
            num_responses_per_prompt=1,
        )
    )

    output_dir: Path | None = None
    continue_on_error: bool = False

    @model_validator(mode="after")
    def _validate_non_empty(self) -> "EvalsConfig":
        if not self.models:
            raise ValueError("EvalsConfig.models must not be empty.")
        if not self.suites:
            raise ValueError("EvalsConfig.suites must not be empty.")
        if self.generation.num_responses_per_prompt != 1:
            raise ValueError(
                "Evals currently supports exactly one response per prompt "
                "(generation.num_responses_per_prompt must be 1)."
            )
        return self


class SuiteEvalResult(BaseModel):
    """Result for one suite run on one model."""

    suite_type: str
    suite_name: str
    model_id: str
    num_samples: int = 0
    artifacts: dict[str, str] = Field(default_factory=dict)
    aggregates: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class ModelEvalResult(BaseModel):
    """Aggregated suite results for one model."""

    model_id: str
    kind: str
    model: str
    adapter_path: str | None = None
    suites: list[SuiteEvalResult] = Field(default_factory=list)


class EvalsResult(BaseModel):
    """Top-level result for an eval run."""

    output_dir: Path | None = None
    num_models: int = 0
    num_suites: int = 0
    num_rows: int = 0
    model_results: list[ModelEvalResult] = Field(default_factory=list)
    leaderboard: list[dict[str, Any]] = Field(default_factory=list)
    summary_path: Path | None = None
