"""Concrete evaluation implementations.

Importing this module registers all built-in evaluations.
"""

from functools import partial

from scripts.persona_metrics.metrics.coherence import CoherenceEvaluation
from scripts.persona_metrics.metrics.counter import CharCounterMetric
from scripts.persona_metrics.metrics.ocean import (
    AgreeablenessEvaluation,
    ConscientiousnessEvaluation,
    ExtraversionEvaluation,
    NeuroticismEvaluation,
    OpennessEvaluation,
)
from scripts.persona_metrics.metrics.text_style import (
    LowercaseDensityEvaluation,
    PunctuationDensityEvaluation,
)
from scripts.persona_metrics.metrics.verb_count import VerbCountEvaluation
from scripts.persona_metrics.registry import register_persona_metric

# Register built-in evaluations
register_persona_metric("agreeableness", AgreeablenessEvaluation)
register_persona_metric("conscientiousness", ConscientiousnessEvaluation)
register_persona_metric("extraversion", ExtraversionEvaluation)
register_persona_metric("neuroticism", NeuroticismEvaluation)
register_persona_metric("openness", OpennessEvaluation)
register_persona_metric("coherence", CoherenceEvaluation)
register_persona_metric("lowercase_density", LowercaseDensityEvaluation)
register_persona_metric("punctuation_density", PunctuationDensityEvaluation)
register_persona_metric("verb_count", VerbCountEvaluation)

# Counter metrics — parameterized instances of CharCounterMetric
register_persona_metric(
    "count_o", partial(CharCounterMetric, metric_name="count_o", target="o")
)

__all__ = [
    "AgreeablenessEvaluation",
    "CharCounterMetric",
    "CoherenceEvaluation",
    "ConscientiousnessEvaluation",
    "ExtraversionEvaluation",
    "LowercaseDensityEvaluation",
    "NeuroticismEvaluation",
    "OpennessEvaluation",
    "PunctuationDensityEvaluation",
    "VerbCountEvaluation",
]
