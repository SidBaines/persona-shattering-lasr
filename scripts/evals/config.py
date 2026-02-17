"""Configuration models for end-to-end eval runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from scripts.common.config import DatasetConfig, GenerationConfig
from scripts.persona_metrics import JudgeLLMConfig, PersonaMetricSpec


# ---------------------------------------------------------------------------
# Shared helpers — used by config validation, run.py, and inspect_bridge.py
# ---------------------------------------------------------------------------


def normalize_component(value: str, fallback: str = "component") -> str:
    """Sanitize a string for use in file paths, metric keys, and suite IDs.

    Replaces non-alphanumeric characters (except ``._-``) with ``_`` and
    strips leading/trailing ``._-``.
    """
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    return normalized.strip("._-") or fallback


def resolve_inspect_task_name(task: str, task_name: str | None) -> str:
    """Derive a human-readable task name from an inspect task reference.

    If *task_name* is explicitly provided it is returned as-is.  Otherwise
    the name is extracted from the task ref string (e.g.
    ``inspect_evals/mmlu_0_shot`` → ``mmlu_0_shot``).
    """
    if task_name:
        return task_name
    if task == "mmlu":
        return "mmlu"
    task_component = task
    if "@" in task_component:
        task_component = task_component.split("@", 1)[1]
    task_component = task_component.rsplit("/", 1)[-1]
    return task_component.replace(".", "_")


def stable_suite_id(suite: "EvalSuiteConfig") -> str:
    """Return a deterministic identifier for a suite configuration.

    Uses the explicit ``suite_id`` when set, otherwise hashes the config
    to produce a short ``auto-<hash>`` identifier.
    """
    if suite.suite_id:
        return normalize_component(suite.suite_id, fallback="suite")
    payload = suite.model_dump(exclude={"suite_id"})
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:10]
    return f"auto-{digest}"


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
    inspect_model: str | None = None

    @model_validator(mode="after")
    def _validate_model_kind(self) -> "EvalModelConfig":
        if self.kind == "lora" and not self.adapter_path:
            raise ValueError("adapter_path is required when kind='lora'.")
        return self


class PersonaMetricsSuiteConfig(BaseModel):
    """Suite config for persona metric scoring on generated responses."""

    type: Literal["persona_metrics"] = "persona_metrics"
    suite_id: str | None = None
    evaluations: list[str | PersonaMetricSpec] = Field(default_factory=lambda: ["count_o"])
    question_column: str | None = "question"
    response_column: str = "response"
    metrics_key: str = "persona_metrics"
    judge: JudgeLLMConfig = Field(default_factory=JudgeLLMConfig)


class InspectTaskSuiteConfig(BaseModel):
    """Suite config for native Inspect task execution."""

    type: Literal["inspect_task"] = "inspect_task"
    suite_id: str | None = None
    # Inspect task ref, e.g. "inspect_evals/mmlu_0_shot" or "path/to/tasks.py@my_task".
    task: str = "mmlu"
    # Extra kwargs forwarded to inspect_ai.eval(...), excluding tasks/model/log_dir.
    eval_kwargs: dict[str, Any] = Field(default_factory=dict)
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
    merged_model_cache_dir: Path = Path("scratch/merged_lora_models")
    force_remerge_lora: bool = False
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
        seen_suite_keys: set[str] = set()
        for suite in self.suites:
            if isinstance(suite, PersonaMetricsSuiteConfig):
                display_name = "persona_metrics"
            else:
                task_name = resolve_inspect_task_name(suite.task, suite.task_name)
                display_name = f"inspect.{task_name}"
            suite_key = f"{display_name}:{stable_suite_id(suite)}"
            if suite_key in seen_suite_keys:
                raise ValueError(
                    "Detected duplicate suite configuration identifier "
                    f"'{suite_key}'. Provide distinct suite_id values."
                )
            seen_suite_keys.add(suite_key)
        return self


class SuiteEvalResult(BaseModel):
    """Result for one suite run on one model."""

    suite_type: str
    suite_name: str
    suite_id: str
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
