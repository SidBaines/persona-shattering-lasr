"""Editing stage configuration."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class RetryConfig(BaseModel):
    """API retry configuration."""

    max_retries: int = 3
    backoff_factor: float = 2.0


class AnthropicProviderConfig(BaseModel):
    """Anthropic-specific settings."""

    max_tokens: int = 1024


class OpenAIProviderConfig(BaseModel):
    """OpenAI-specific settings."""

    model: str | None = None  # Override model for OpenAI (if different from main model)
    max_tokens: int = 1024


class QualityConfig(BaseModel):
    """Edit quality evaluation configuration."""

    enabled: bool = True
    metrics: list[str] = ["count_o"]
    reporters: list[str] = ["json"]
    metrics_key: str = "quality_metrics"


class EditingConfig(BaseModel):
    """Configuration for the editing stage.

    Example:
        config = EditingConfig(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            prompt_template="default_persona_shatter",
            max_concurrent=10,
            output_path=Path("scratch/edited_dataset.jsonl"),
        )
        dataset, result = run_editing(config, input_dataset)
    """

    # Provider settings
    provider: str = "anthropic"  # "anthropic" or "openai"
    model: str = "claude-sonnet-4-20250514"
    prompt_template: str = "default_persona_shatter"

    # Concurrency and timeout
    max_concurrent: int = 10
    timeout: int = 60

    # Retry settings
    retry: RetryConfig = RetryConfig()

    # Provider-specific settings
    anthropic: AnthropicProviderConfig = AnthropicProviderConfig()
    openai: OpenAIProviderConfig = OpenAIProviderConfig()

    # Quality metrics
    quality: QualityConfig = QualityConfig()

    # Output
    output_path: Path | None = None  # If None, returns dataset without saving


class EditingResult(BaseModel):
    """Result from running editing."""

    class Config:
        arbitrary_types_allowed = True

    output_path: Path | None = None
    num_samples: int = 0
    num_failed: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
