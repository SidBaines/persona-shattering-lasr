"""Persona metrics module for running metrics on datasets at any pipeline stage.

Example:
    from scripts.persona_metrics import run_persona_metrics, PersonaMetricsConfig

    config = PersonaMetricsConfig(
        evaluations=["count_o", "coherence"],
        response_column="response",
    )
    dataset, result = run_persona_metrics(config, dataset=my_dataset)
"""

from scripts.persona_metrics.aggregation import aggregate_persona_metric_results
from scripts.persona_metrics.base import PersonaMetric, PersonaMetricContext
from scripts.persona_metrics.config import (
    PersonaMetricsConfig,
    PersonaMetricsResult,
    PersonaMetricSpec,
    JudgeLLMConfig,
)
from scripts.persona_metrics.registry import (
    PERSONA_METRIC_REGISTRY,
    get_persona_metric,
    register_persona_metric,
)
from scripts.persona_metrics.run import (
    create_persona_metrics,
    run_persona_metrics,
    run_persona_metrics_async,
)

# Import metrics subpackage to trigger registration of built-ins
import scripts.persona_metrics.metrics  # noqa: F401

__all__ = [
    "PersonaMetric",
    "PersonaMetricContext",
    "PersonaMetricsConfig",
    "PersonaMetricsResult",
    "PersonaMetricSpec",
    "JudgeLLMConfig",
    "PERSONA_METRIC_REGISTRY",
    "get_persona_metric",
    "register_persona_metric",
    "run_persona_metrics",
    "run_persona_metrics_async",
    "create_persona_metrics",
    "aggregate_persona_metric_results",
]
