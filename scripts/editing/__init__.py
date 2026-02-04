"""Editing module for API-based response editing with quality tracking."""

from scripts.editing.prompts import TEMPLATES, get_prompt
from scripts.editing.anthropic_client import edit_response as anthropic_edit
from scripts.editing.openai_client import edit_response as openai_edit
from scripts.editing.quality import (
    EditQualityMetric,
    CountOMetric,
    get_metric,
    aggregate_metrics,
    QualityReporter,
    JsonReporter,
    get_reporters,
)

# Import run_editing after other imports to avoid circular imports
# (run.py imports from this module)
def _get_run_editing():
    from scripts.editing.run import run_editing
    return run_editing

__all__ = [
    "TEMPLATES",
    "get_prompt",
    "anthropic_edit",
    "openai_edit",
    "EditQualityMetric",
    "CountOMetric",
    "get_metric",
    "aggregate_metrics",
    "QualityReporter",
    "JsonReporter",
    "get_reporters",
    "run_editing",
]

def __getattr__(name):
    if name == "run_editing":
        return _get_run_editing()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
