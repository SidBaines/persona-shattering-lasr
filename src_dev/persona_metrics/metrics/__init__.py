"""Concrete evaluation implementations.

Importing this module registers all built-in evaluations.
"""

from functools import partial

from src_dev.persona_metrics.metrics.coherence import (
    BetterCoherenceEvaluation,
    CoherenceEvaluation,
    CoherenceV2Evaluation,
)
from src_dev.persona_metrics.metrics.counter import CharCounterMetric
from src_dev.persona_metrics.metrics.ocean import (
    AgreeablenessEvaluation,
    ConscientiousnessEvaluation,
    ExtraversionEvaluation,
    NeuroticismEvaluation,
    OpennessEvaluation,
)
from src_dev.persona_metrics.metrics.ocean_v2 import (
    AgreeablenessV2Evaluation,
    ConscientiousnessV2Evaluation,
    ExtraversionV2Evaluation,
    NeuroticismV2Evaluation,
    OpennessV2Evaluation,
)
from src_dev.persona_metrics.metrics.text_style import (
    LowercaseDensityEvaluation,
    PunctuationDensityEvaluation,
)
from src_dev.persona_metrics.metrics.verb_count import VerbCountEvaluation
from src_dev.persona_metrics.registry import register_persona_metric

# ── Current judges (use these) ──
register_persona_metric("agreeableness_v2", AgreeablenessV2Evaluation)
register_persona_metric("conscientiousness_v2", ConscientiousnessV2Evaluation)
register_persona_metric("extraversion_v2", ExtraversionV2Evaluation)
register_persona_metric("neuroticism_v2", NeuroticismV2Evaluation)
register_persona_metric("openness_v2", OpennessV2Evaluation)
register_persona_metric("coherence_v2", CoherenceV2Evaluation)

# ── Deprecated judges (kept for backward compatibility) ──
register_persona_metric("agreeableness", AgreeablenessEvaluation)
register_persona_metric("conscientiousness", ConscientiousnessEvaluation)
register_persona_metric("extraversion", ExtraversionEvaluation)
register_persona_metric("neuroticism", NeuroticismEvaluation)
register_persona_metric("openness", OpennessEvaluation)
register_persona_metric("coherence", CoherenceEvaluation)
register_persona_metric("better_coherence_judge", BetterCoherenceEvaluation)
register_persona_metric("lowercase_density", LowercaseDensityEvaluation)
register_persona_metric("punctuation_density", PunctuationDensityEvaluation)
register_persona_metric("verb_count", VerbCountEvaluation)

# Counter metrics — parameterized instances of CharCounterMetric
register_persona_metric(
    "count_o", partial(CharCounterMetric, metric_name="count_o", target="o")
)
register_persona_metric(
    "count_t", partial(CharCounterMetric, metric_name="count_t", target="t")
)

__all__ = [
    "AgreeablenessEvaluation",
    "AgreeablenessV2Evaluation",
    "CharCounterMetric",
    "BetterCoherenceEvaluation",
    "CoherenceEvaluation",
    "CoherenceV2Evaluation",
    "ConscientiousnessEvaluation",
    "ConscientiousnessV2Evaluation",
    "ExtraversionEvaluation",
    "ExtraversionV2Evaluation",
    "LowercaseDensityEvaluation",
    "NeuroticismEvaluation",
    "NeuroticismV2Evaluation",
    "OpennessEvaluation",
    "OpennessV2Evaluation",
    "PunctuationDensityEvaluation",
    "VerbCountEvaluation",
]
