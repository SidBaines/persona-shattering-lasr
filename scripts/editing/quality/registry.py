"""Registry for edit quality metrics.

Provides a central place to register and retrieve metric implementations.
"""

from __future__ import annotations

from scripts.editing.quality.metrics import LevelOfPersonaMetric, EditQualityMetric

# Global registry mapping metric names to their classes
METRIC_REGISTRY: dict[str, type[EditQualityMetric]] = {
    "level_of_persona": LevelOfPersonaMetric,
}


def get_metric(name: str) -> EditQualityMetric:
    """Get a metric instance by name.

    Args:
        name: Metric name (must be registered).

    Returns:
        Instantiated metric.

    Raises:
        KeyError: If metric name is not registered.
    """
    if name not in METRIC_REGISTRY:
        available = ", ".join(sorted(METRIC_REGISTRY.keys()))
        raise KeyError(f"Unknown metric '{name}'. Available metrics: {available}")
    return METRIC_REGISTRY[name]()


def register_metric(name: str, metric_class: type[EditQualityMetric]) -> None:
    """Register a custom metric.

    Args:
        name: Unique name for the metric.
        metric_class: Class implementing EditQualityMetric protocol.

    Raises:
        ValueError: If name is already registered.
    """
    if name in METRIC_REGISTRY:
        raise ValueError(f"Metric '{name}' is already registered")
    METRIC_REGISTRY[name] = metric_class
