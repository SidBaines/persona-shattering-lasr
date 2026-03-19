"""OCEAN Big Five personality trait evaluations using an LLM judge.

.. deprecated::
    These v1 classes use ad-hoc judge_configs templates and a vague scale.
    Prefer the v2 equivalents in ``ocean_v2.py``, which are built from the
    canonical OCEAN_DEFINITION and use a calibrated -4...+4 ordinal scale:

        agreeableness     → AgreeablenessV2Evaluation     ("agreeableness_v2")
        conscientiousness → ConscientiousnessV2Evaluation  ("conscientiousness_v2")
        extraversion      → ExtraversionV2Evaluation       ("extraversion_v2")
        neuroticism       → NeuroticismV2Evaluation        ("neuroticism_v2")
        openness          → OpennessV2Evaluation            ("openness_v2")

    The v1 classes remain registered for backward compatibility and for
    comparison runs that quantify the improvement from v1 to v2.
"""

from __future__ import annotations

from src_dev.persona_metrics.metrics.judge_configs import (
    AGREEABLENESS_EXAMPLES,
    CONSCIENTIOUSNESS_EXAMPLES,
    DEFAULT_AGREEABLENESS_TEMPLATE,
    DEFAULT_CONSCIENTIOUSNESS_TEMPLATE,
    DEFAULT_EXTRAVERSION_TEMPLATE,
    DEFAULT_NEUROTICISM_TEMPLATE,
    DEFAULT_OPENNESS_TEMPLATE,
    EXTRAVERSION_EXAMPLES,
    NEUROTICISM_EXAMPLES,
    OPENNESS_EXAMPLES,
)
from src_dev.persona_metrics.metrics.llm_judge_base import (
    LLMJudgeMetric,
    _parse_judge_response,
)

__all__ = [
    "AgreeablenessEvaluation",
    "ConscientiousnessEvaluation",
    "ExtraversionEvaluation",
    "NeuroticismEvaluation",
    "OpennessEvaluation",
    "_parse_judge_response",  # re-exported for test compatibility
]


class AgreeablenessEvaluation(LLMJudgeMetric):
    name = "agreeableness"
    default_template = DEFAULT_AGREEABLENESS_TEMPLATE
    default_examples = AGREEABLENESS_EXAMPLES


class ConscientiousnessEvaluation(LLMJudgeMetric):
    name = "conscientiousness"
    default_template = DEFAULT_CONSCIENTIOUSNESS_TEMPLATE
    default_examples = CONSCIENTIOUSNESS_EXAMPLES


class ExtraversionEvaluation(LLMJudgeMetric):
    name = "extraversion"
    default_template = DEFAULT_EXTRAVERSION_TEMPLATE
    default_examples = EXTRAVERSION_EXAMPLES


class NeuroticismEvaluation(LLMJudgeMetric):
    name = "neuroticism"
    default_template = DEFAULT_NEUROTICISM_TEMPLATE
    default_examples = NEUROTICISM_EXAMPLES


class OpennessEvaluation(LLMJudgeMetric):
    name = "openness"
    default_template = DEFAULT_OPENNESS_TEMPLATE
    default_examples = OPENNESS_EXAMPLES
