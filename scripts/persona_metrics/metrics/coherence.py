"""Coherence evaluation: LLM-as-judge for response coherence scoring."""

from __future__ import annotations

from scripts.persona_metrics.metrics.judge_configs import (
    COHERENCE_EXAMPLES,
    DEFAULT_COHERENCE_TEMPLATE,
)
from scripts.persona_metrics.metrics.llm_judge_base import LLMJudgeMetric


class CoherenceEvaluation(LLMJudgeMetric):
    name = "coherence"
    default_template = DEFAULT_COHERENCE_TEMPLATE
    default_examples = COHERENCE_EXAMPLES
    score_min = 0
    score_max = 100
    score_default = 50
    score_error = -1
