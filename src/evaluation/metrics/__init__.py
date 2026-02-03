"""Metric implementations."""

from ..base import Metric
from .llm_judge import LLMJudgeMetric
from .simple import CharCountMetric

METRICS: dict[str, type[Metric]] = {
    "count_char": CharCountMetric,
    "llm_judge": LLMJudgeMetric,
}


def get_metric(metric_type: str) -> Metric:
    """Get a metric by type.

    Args:
        metric_type: Type of metric (e.g., "count_char", "llm_judge").

    Returns:
        An instance of the requested metric.

    Raises:
        KeyError: If metric_type is not registered.
    """
    if metric_type not in METRICS:
        raise KeyError(f"Unknown metric type: {metric_type}. Available: {list(METRICS.keys())}")
    return METRICS[metric_type]()
