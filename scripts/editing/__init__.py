"""Editing module for LLM- or code-based response editing with quality tracking.

Example:
    from scripts.editing import run_editing, EditingConfig

    config = EditingConfig(
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        prompt_template="default_persona_shatter",
        output_path=Path("scratch/edited.jsonl"),
    )
    dataset, result = run_editing(config, input_dataset)
"""

from scripts.editing.config import (
    EditingConfig,
    EditingResult,
    RetryConfig,
    AnthropicProviderConfig,
    OpenAIProviderConfig,
    CodeProviderConfig,
    QualityConfig,
)
from scripts.editing.prompts import TEMPLATES, get_prompt
from scripts.editing.quality import (
    EditQualityMetric,
    LevelOfPersonaMetric,
    get_metric,
    aggregate_metrics,
    QualityReporter,
    JsonReporter,
    get_reporters,
)

# Import run_editing after other imports to avoid circular imports
def _get_run_editing():
    from scripts.editing.run import run_editing
    return run_editing

def _get_main():
    from scripts.editing.cli import main
    return main

__all__ = [
    # Config classes
    "EditingConfig",
    "EditingResult",
    "RetryConfig",
    "AnthropicProviderConfig",
    "OpenAIProviderConfig",
    "CodeProviderConfig",
    "QualityConfig",
    # Prompts
    "TEMPLATES",
    "get_prompt",
    # Quality
    "EditQualityMetric",
    "LevelOfPersonaMetric",
    "get_metric",
    "aggregate_metrics",
    "QualityReporter",
    "JsonReporter",
    "get_reporters",
    # Run function
    "run_editing",
    # CLI entry point
    "main",
]

def __getattr__(name):
    if name == "run_editing":
        return _get_run_editing()
    if name == "main":
        return _get_main()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
