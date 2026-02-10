"""Aggregation utilities for evaluation results."""

from __future__ import annotations

import statistics
from collections import defaultdict


def aggregate_evaluation_results(
    all_record_results: list[dict[str, float | int | str]],
) -> dict[str, float]:
    """Aggregate evaluation results across all records.

    Computes mean, median, min, max, and stdev for each numeric metric.
    String-valued metrics are skipped.

    Args:
        all_record_results: List of result dicts from each record.

    Returns:
        Dict with keys like "{metric_name}.mean", "{metric_name}.median", etc.
    """
    if not all_record_results:
        return {}

    values_by_metric: dict[str, list[float]] = defaultdict(list)
    for record_results in all_record_results:
        for key, value in record_results.items():
            if isinstance(value, (int, float)):
                values_by_metric[key].append(float(value))

    aggregates: dict[str, float] = {}
    for metric_name, values in sorted(values_by_metric.items()):
        if not values:
            continue
        aggregates[f"{metric_name}.mean"] = statistics.mean(values)
        aggregates[f"{metric_name}.median"] = statistics.median(values)
        aggregates[f"{metric_name}.min"] = min(values)
        aggregates[f"{metric_name}.max"] = max(values)
        if len(values) >= 2:
            aggregates[f"{metric_name}.stdev"] = statistics.stdev(values)
        else:
            aggregates[f"{metric_name}.stdev"] = 0.0

    return aggregates
