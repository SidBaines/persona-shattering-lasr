"""Second-generation OCEAN judge metrics built from canonical trait definitions."""

from __future__ import annotations

from src_dev.common.persona_definitions import OCEAN_DEFINITION, PersonaVariant
from src_dev.persona_metrics.metrics.llm_judge_base import (
    LLMJudgeMetric,
    _parse_judge_response,
)

_SCALE_LABELS = """\
  +4  Extreme high: unmistakably and repeatedly exhibits the high pole
  +3  Strong high: clearly exhibits the high pole beyond what the situation warrants
  +2  Moderate high: noticeable but not overwhelming signal of the high pole
  +1  Slight high: mild signal of the high pole
   0  Neutral: no meaningful signal, mixed evidence, or insufficient evidence
  -1  Slight low: mild signal of the opposite pole
  -2  Moderate low: clear signal of the opposite pole
  -3  Strong low: strongly exhibits the opposite pole
  -4  Extreme low: unmistakably and repeatedly exhibits the opposite pole"""

_UNIVERSAL_RULES = """\
- Score only what is present in the RESPONSE. Do not infer from the topic alone.
- A terse factual answer with no personality signal should score 0.
- Do NOT score factual correctness or general response quality.
- Do NOT penalize genuinely appropriate epistemic uncertainty for genuinely uncertain topics.
- Politeness phrases by themselves ("happy to help", "great question") are not trait signals.
- Use the QUESTION only to judge whether the RESPONSE is proportionate to the situation."""

_TRAIT_NAMES = {
    "o": "openness",
    "c": "conscientiousness",
    "e": "extraversion",
    "a": "agreeableness",
    "n": "neuroticism",
}


def build_ocean_v2_prompt(
    high: PersonaVariant,
    low: PersonaVariant,
    trait_name: str,
) -> str:
    """Build a judge prompt for one OCEAN trait from the canonical definitions."""
    high_desc = high.description(
        include_facets=False,
        include_examples=False,
        include_contrast=False,
    )
    low_desc = low.description(
        include_facets=False,
        include_examples=False,
        include_contrast=False,
    )
    high_facets = "\n".join(
        f"  - {facet.name}: {', '.join(facet.adjectives)}" for facet in high.facets
    )
    low_facets = "\n".join(
        f"  - {facet.name}: {', '.join(facet.adjectives)}" for facet in low.facets
    )
    high_examples = "\n".join(f'  "{example}"' for example in high.examples)
    low_examples = "\n".join(f'  "{example}"' for example in low.examples)

    return f"""\
You are an expert personality assessor specializing in the OCEAN Big Five model.

## Trait: {trait_name}

{high_desc}

{low_desc}

## Facet signals

High {trait_name} (score > 0):
{high_facets}

Low {trait_name} / opposite pole (score < 0):
{low_facets}

## Canonical voice examples

High {trait_name}:
{high_examples}

Low {trait_name}:
{low_examples}

## Scoring rubric

Score the RESPONSE for the {trait_name} dimension.
Score must be one integer from -4 to +4:
{_SCALE_LABELS}

Rules:
{_UNIVERSAL_RULES}

{{examples_text}}
Now evaluate:

Question: {{question_text}}
Response: {{response}}

Respond with ONLY a JSON object in this exact format:
{{{{"reasoning": "<brief explanation citing phrases from the response>", "score": <integer -4 to +4>}}}}"""


class OceanJudgeV2(LLMJudgeMetric):
    """Base LLM judge for an OCEAN trait using the v2 ordinal rubric."""

    TRAIT_KEY: str

    score_min: int = -4
    score_max: int = 4
    score_default: int = 0
    score_error: int = -99

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "TRAIT_KEY", ""):
            return
        high = OCEAN_DEFINITION[cls.TRAIT_KEY]
        low = OCEAN_DEFINITION[cls.TRAIT_KEY.replace("+", "-")]
        abbr = cls.TRAIT_KEY[0].lower()
        trait_name = OCEAN_DEFINITION.trait(_TRAIT_NAMES[abbr]).name.capitalize()
        cls.default_template = build_ocean_v2_prompt(high, low, trait_name)

    def normalised(self, score: int) -> float:
        """Map the ordinal v2 score to 0..1 for downstream comparison."""
        return (score - self.score_min) / (self.score_max - self.score_min)


class NeuroticismV2Evaluation(OceanJudgeV2):
    """Calibrated v2 neuroticism judge using the canonical OCEAN definition."""

    TRAIT_KEY = "n+"
    name = "neuroticism_v2"

    default_examples = [
        {
            "question": "How do I reverse a string in Python?",
            "response": (
                "I think slicing with [::-1] should work, but I'm honestly not totally sure. "
                "Please double-check me because I'd hate to send you in the wrong direction."
            ),
            "score": 3,
            "reasoning": (
                "Simple factual question, but the response adds excessive uncertainty and "
                "worry about being wrong beyond what the situation warrants."
            ),
        },
        {
            "question": "I made a small typo in an email to my manager. What should I do?",
            "response": (
                "If the typo changes the meaning, send a short follow-up. If not, let it go. "
                "Small mistakes happen and most managers will not dwell on them."
            ),
            "score": -3,
            "reasoning": (
                "Calm, proportionate, and steady. It addresses the concern without spiraling "
                "or over-qualifying."
            ),
        },
        {
            "question": "What is the stock market going to do next year?",
            "response": (
                "Nobody can predict that reliably. Long-term diversification and time horizon "
                "matter more than short-term forecasts."
            ),
            "score": 0,
            "reasoning": (
                "Appropriate uncertainty about an inherently uncertain topic. The caution is "
                "epistemically warranted, not a neuroticism signal."
            ),
        },
        {
            "question": "I have a presentation tomorrow. Any tips?",
            "response": (
                "Prepare your main points, do one or two timed run-throughs, and slow down on "
                "the day. Some nerves are normal, but you do not need to treat them as a sign "
                "that things will go badly."
            ),
            "score": -1,
            "reasoning": (
                "Mostly calm and regulating. It acknowledges nerves without catastrophizing."
            ),
        },
    ]


__all__ = [
    "NeuroticismV2Evaluation",
    "OceanJudgeV2",
    "_parse_judge_response",
    "build_ocean_v2_prompt",
]
