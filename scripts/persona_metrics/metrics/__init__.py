"""Concrete evaluation implementations.

Importing this module registers all built-in evaluations.
"""

from scripts.persona_metrics.metrics.coherence import CoherenceEvaluation
from scripts.persona_metrics.metrics.count_o import OCountEvaluation
from scripts.persona_metrics.metrics.count_p import PCountEvaluation
from scripts.persona_metrics.metrics.neuroticism import NeuroticismEvaluation
from scripts.persona_metrics.metrics.text_style import (
    LowercaseDensityEvaluation,
    PunctuationDensityEvaluation,
)
from scripts.persona_metrics.metrics.verb_count import VerbCountEvaluation
from scripts.persona_metrics.registry import register_persona_metric

# Register built-in evaluations
register_persona_metric("count_o", OCountEvaluation)
register_persona_metric("count_p", PCountEvaluation)
register_persona_metric("verb_count", VerbCountEvaluation)
register_persona_metric("coherence", CoherenceEvaluation)
register_persona_metric("neuroticism", NeuroticismEvaluation)
register_persona_metric("lowercase_density", LowercaseDensityEvaluation)
register_persona_metric("punctuation_density", PunctuationDensityEvaluation)

__all__ = [
    "OCountEvaluation",
    "PCountEvaluation",
    "VerbCountEvaluation",
    "CoherenceEvaluation",
    "NeuroticismEvaluation",
    "LowercaseDensityEvaluation",
    "PunctuationDensityEvaluation",
]
