"""OCEAN Big Five personality trait evaluations using an LLM judge."""

from __future__ import annotations

from scripts.persona_metrics.metrics.judge_configs import (
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
from scripts.persona_metrics.metrics.llm_judge_base import (
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
