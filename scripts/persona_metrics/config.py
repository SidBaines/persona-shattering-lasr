"""Evaluation stage configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from scripts.common.config import DatasetConfig


class JudgeLLMConfig(BaseModel):
    """Configuration for LLM-as-judge evaluations (e.g., CoherenceEvaluation)."""

    provider: str = "openai"  # "openai", "openrouter", "anthropic"
    model: str = "gpt-4o-mini"
    api_key_env: str | None = None  # If None, uses default for provider
    max_tokens: int = 1024
    temperature: float = 0.0  # Deterministic by default for judging
    max_concurrent: int = 10
    timeout: int = 60


class PersonaMetricSpec(BaseModel):
    """Specification for a single evaluation with optional per-evaluation params.

    Use this when you need to pass evaluation-specific configuration such as
    custom thresholds, prompt templates, or a different judge model.

    Example:
        PersonaMetricSpec(name="coherence", params={"judge_config": JudgeLLMConfig(model="gpt-4o")})
        PersonaMetricSpec(name="regex_match", params={"pattern": r"\\b(safe|unsafe)\\b"})
    """

    name: str
    params: dict[str, Any] = {}


class PersonaMetricsConfig(BaseModel):
    """Configuration for the evaluation stage.

    Example:
        config = PersonaMetricsConfig(
            evaluations=["count_o", "coherence"],
            dataset=DatasetConfig(source="local", path="scratch/inference_output.jsonl"),
            response_column="response",
            question_column="question",
            output_path=Path("scratch/evaluation_results.jsonl"),
        )
        dataset, result = run_persona_metrics(config)

        # With per-evaluation params:
        config = PersonaMetricsConfig(
            evaluations=[
                "count_o",
                PersonaMetricSpec(name="coherence", params={"judge_config": JudgeLLMConfig(model="gpt-4o")}),
            ],
        )
    """

    # Which evaluations to run (strings or PersonaMetricSpec for per-eval config)
    evaluations: list[str | PersonaMetricSpec] = ["count_o"]

    # Dataset settings (for standalone run; not needed when called inline)
    dataset: DatasetConfig = DatasetConfig()

    # Column mapping
    response_column: str = "response"
    question_column: str | None = "question"

    # LLM judge settings (for evaluations that need an LLM)
    judge: JudgeLLMConfig = JudgeLLMConfig()

    # Key under which per-record metrics are embedded in the dataset
    metrics_key: str = "persona_metrics"

    # Output
    output_path: Path | None = None


class PersonaMetricsResult(BaseModel):
    """Result from running evaluation."""

    class Config:
        arbitrary_types_allowed = True

    output_path: Path | None = None
    num_samples: int = 0
    evaluations_run: list[str] = []
    aggregates: dict[str, Any] = {}
