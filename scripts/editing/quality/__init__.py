"""Edit quality evaluation infrastructure.

This module provides metrics, reporters, and aggregation for evaluating
the quality of LLM edits during the editing stage.
"""

from scripts.editing.quality.aggregation import aggregate_metrics
from scripts.editing.quality.metrics import CountOMetric, EditQualityMetric, PassiveVoiceMetric
from scripts.editing.quality.registry import (
    METRIC_REGISTRY,
    get_metric,
    register_metric,
)
from scripts.editing.quality.reporters import (
    REPORTER_REGISTRY,
    JsonReporter,
    QualityReporter,
    get_reporters,
)

__all__ = [
    # Metrics
    "EditQualityMetric",
    "CountOMetric",
    "PassiveVoiceMetric",
    # Registry
    "METRIC_REGISTRY",
    "get_metric",
    "register_metric",
    # Reporters
    "QualityReporter",
    "JsonReporter",
    "REPORTER_REGISTRY",
    "get_reporters",
    # Aggregation
    "aggregate_metrics",
]
