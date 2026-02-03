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
]
