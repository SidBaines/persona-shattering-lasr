"""Coherence evaluation: LLM-as-judge for response coherence scoring."""

from __future__ import annotations

from src_dev.persona_metrics.metrics.judge_configs import (
    COHERENCE_EXAMPLES,
    DEFAULT_COHERENCE_TEMPLATE,
    BETTER_COHERENCE_EXAMPLES, 
    BETTER_DEFAULT_COHERENCE_TEMPLATE
)
from src_dev.persona_metrics.metrics.llm_judge_base import LLMJudgeMetric


class CoherenceEvaluation(LLMJudgeMetric):
    name = "coherence"
    default_template = DEFAULT_COHERENCE_TEMPLATE
    default_examples = COHERENCE_EXAMPLES
    score_min = 0
    score_max = 100
    score_default = 50
    score_error = -1


class BetterCoherenceEvaluation(LLMJudgeMetric):
    name = "better_coherence_judge"
    default_template = BETTER_DEFAULT_COHERENCE_TEMPLATE
    default_examples = BETTER_COHERENCE_EXAMPLES
    score_min = 0
    score_max = 10
    score_default = 5
    score_error = -1
