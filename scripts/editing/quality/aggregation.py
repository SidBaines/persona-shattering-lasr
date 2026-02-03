"""Aggregation utilities for quality metrics.

Computes summary statistics across all records.
"""

from __future__ import annotations

import statistics
from collections import defaultdict


def aggregate_metrics(all_record_metrics: list[dict[str, float | int]]) -> dict[str, float]:
    """Aggregate metrics across all records.

    Computes mean, median, min, max, and stdev for each numeric metric.

    Args:
        all_record_metrics: List of metric dicts from each record.

    Returns:
        Dict with keys like "{metric_name}.mean", "{metric_name}.median", etc.
    """
    if not all_record_metrics:
        return {}

    # Group values by metric name
    values_by_metric: dict[str, list[float]] = defaultdict(list)
    for record_metrics in all_record_metrics:
        for key, value in record_metrics.items():
            if isinstance(value, (int, float)):
                values_by_metric[key].append(float(value))

    # Compute aggregates for each metric
    aggregates: dict[str, float] = {}
    for metric_name, values in values_by_metric.items():
        if not values:
            continue

        aggregates[f"{metric_name}.mean"] = statistics.mean(values)
        aggregates[f"{metric_name}.median"] = statistics.median(values)
        aggregates[f"{metric_name}.min"] = min(values)
        aggregates[f"{metric_name}.max"] = max(values)

        # stdev requires at least 2 values
        if len(values) >= 2:
            aggregates[f"{metric_name}.stdev"] = statistics.stdev(values)
        else:
            aggregates[f"{metric_name}.stdev"] = 0.0

    return aggregates
