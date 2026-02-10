"""Evaluation stage configuration."""

from __future__ import annotations

from pathlib import Path

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


class EvaluationConfig(BaseModel):
    """Configuration for the evaluation stage.

    Example:
        config = EvaluationConfig(
            evaluations=["count_o", "coherence"],
            dataset=DatasetConfig(source="local", path="scratch/inference_output.jsonl"),
            response_column="response",
            question_column="question",
            output_path=Path("scratch/evaluation_results.jsonl"),
        )
        dataset, result = run_evaluation(config)
    """

    # Which evaluations to run
    evaluations: list[str] = ["count_o"]

    # Dataset settings (for standalone run; not needed when called inline)
    dataset: DatasetConfig = DatasetConfig()

    # Column mapping
    response_column: str = "response"
    question_column: str | None = "question"

    # LLM judge settings (for evaluations that need an LLM)
    judge: JudgeLLMConfig = JudgeLLMConfig()

    # Key under which per-record metrics are embedded in the dataset
    metrics_key: str = "evaluation_metrics"

    # Output
    output_path: Path | None = None


class EvaluationResult(BaseModel):
    """Result from running evaluation."""

    class Config:
        arbitrary_types_allowed = True

    output_path: Path | None = None
    num_samples: int = 0
    evaluations_run: list[str] = []
    aggregates: dict[str, float] = {}
