"""Bridge between lm_eval's ``process_results`` hook and persona_metrics.

Each ``process_results_<name>`` function is referenced by the corresponding
YAML task config via ``process_results: !function utils.process_results_<name>``.

lm_eval calls these with ``(doc, results)`` after generation:
- ``doc``: the dataset row as a dict
- ``results``: list of generated strings (length 1 for generate_until)

The function must return ``dict[str, float | int]`` mapping metric names
(matching ``metric_list`` in the YAML) to numeric values.
"""

from __future__ import annotations

from scripts.persona_metrics.base import PersonaMetricContext
from scripts.persona_metrics.registry import get_persona_metric

_metric_cache: dict = {}


def _get_metric(name: str):
    if name not in _metric_cache:
        # Trigger registration of built-in metrics
        import scripts.persona_metrics.metrics  # noqa: F401

        _metric_cache[name] = get_persona_metric(name)
    return _metric_cache[name]


def _run_metric(metric_name: str, doc: dict, results: list[str]) -> dict:
    response = results[0] if results else ""
    question = doc.get("question", "")
    metric = _get_metric(metric_name)
    context = PersonaMetricContext(
        response=response, question=question, record=dict(doc)
    )
    raw = metric.evaluate(response, question, context=context)
    # Filter to numeric values only — lm_eval aggregation cannot handle strings
    return {k: v for k, v in raw.items() if isinstance(v, (int, float))}


def process_results_count_o(doc: dict, results: list[str]) -> dict:
    return _run_metric("count_o", doc, results)


def process_results_verb_count(doc: dict, results: list[str]) -> dict:
    return _run_metric("verb_count", doc, results)


def process_results_coherence(doc: dict, results: list[str]) -> dict:
    return _run_metric("coherence", doc, results)


def process_results_lowercase_density(doc: dict, results: list[str]) -> dict:
    return _run_metric("lowercase_density", doc, results)


def process_results_punctuation_density(doc: dict, results: list[str]) -> dict:
    return _run_metric("punctuation_density", doc, results)
