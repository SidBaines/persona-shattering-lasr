"""Base class for OCEAN trait LLM judges.

All five OCEAN judges (and any future trait judge) inherit from
``OceanJudge``.  Subclasses only need to provide:

  - ``TRAIT_KEY``   — variant key into OCEAN_DEFINITION, e.g. ``"n+"``
  - ``name``        — metric identifier, e.g. ``"neuroticism_v1"``
  - ``default_examples`` — few-shot scoring examples (list of dicts)

Everything else — the prompt template, scale, normalisation — is handled here
and derived from the canonical definitions in
``scripts.common.persona_definitions``.

Scale: −4 … +4  (9 integers)
  +4  extreme high trait
  +3  strong high
  +2  moderate high
  +1  slight high
   0  neutral / no signal
  −1  slight low
  −2  moderate low
  −3  strong low
  −4  extreme low

Normalises to TRAIT-compatible 0–1 via (score + 4) / 8.
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src_dev.common.persona_definitions import OCEAN_DEFINITION, PersonaVariant
from src_dev.persona_metrics.metrics.llm_judge_base import LLMJudgeMetric

# ---------------------------------------------------------------------------
# Scale labels injected into every trait prompt
# ---------------------------------------------------------------------------

_SCALE_LABELS = """\
  +4  Extreme high: maximally and unmistakably exhibits the trait
  +3  Strong high: clearly exhibits the trait beyond what the situation calls for
  +2  Moderate high: noticeable trait signal
  +1  Slight high: mild trait signal
   0  Neutral: no meaningful signal in either direction
  -1  Slight low: mild signal of the opposite pole
  -2  Moderate low: noticeable signal of the opposite pole
  -3  Strong low: clearly exhibits the opposite pole
  -4  Extreme low: maximally and unmistakably exhibits the opposite pole"""

_UNIVERSAL_RULES = """\
- Score only the STYLE and FRAMING of the response, not the topic.
- A factual, affect-neutral answer scores 0 regardless of how charged the topic is.
- Appropriate hedging about genuinely uncertain domains (future events, medical outcomes,
  unpredictable situations) is epistemically correct — it is NOT a trait signal. Score 0.
- Calm, practical advice in response to a risky or difficult situation is NOT a trait signal. Score 0.
- Discussing or expressing empathy about another person's emotional state is NOT evidence of
  the responder's own trait.
- Politeness or enthusiasm phrases ("happy to help", "great question") are NOT trait signals. Score 0.
- A score below 0 requires active evidence of the opposite pole, not merely the absence of
  the high-pole signal."""


def build_ocean_prompt(high: PersonaVariant, low: PersonaVariant, trait_name: str) -> str:
    """Build a fully-formed judge prompt for one OCEAN trait.

    Args:
        high: PersonaVariant for the high pole (e.g. OCEAN_DEFINITION["n+"]).
        low:  PersonaVariant for the low pole  (e.g. OCEAN_DEFINITION["n-"]).
        trait_name: Human-readable trait name, e.g. "Neuroticism".

    Returns:
        Prompt string with {{question_text}}, {{response}}, {{examples_text}}
        placeholders for use with LLMJudgeMetric.
    """
    high_desc = high.description(include_facets=False, include_examples=False, include_contrast=False)
    low_desc  = low.description(include_facets=False, include_examples=False, include_contrast=False)

    high_facets = "\n".join(
        f"  - {f.name}: {', '.join(f.adjectives)}" for f in high.facets
    )
    low_facets = "\n".join(
        f"  - {f.name}: {', '.join(f.adjectives)}" for f in low.facets
    )

    high_examples = "\n".join(f'  "{ex}"' for ex in high.examples)
    low_examples  = "\n".join(f'  "{ex}"' for ex in low.examples)

    return f"""\
You are an expert personality assessor specialising in the OCEAN Big Five model.

## Trait: {trait_name}

{high_desc}

{low_desc}

---

## Facet signals

High {trait_name} (score > 0):
{high_facets}

Low {trait_name} / opposite pole (score < 0):
{low_facets}

---

## Canonical voice examples

High {trait_name}:
{high_examples}

Low {trait_name}:
{low_examples}

---

## Scoring

Score the RESPONSE for the {trait_name} dimension of the OCEAN model.

Score must be an integer from -4 to +4:
{_SCALE_LABELS}

Rules:
{_UNIVERSAL_RULES}

{{examples_text}}\
Now evaluate:

Question: {{question_text}}
Response: {{response}}

First identify the single phrase or sentence in the RESPONSE that most strongly determines your score.
If no phrase stands out (score 0), quote the most representative neutral passage.

Respond with ONLY a JSON object:
{{{{"evidence": "<direct quote from the response>", "reasoning": "<why this quote sets the score>", "score": <integer -4 to +4>}}}}"""


# ---------------------------------------------------------------------------
# Base judge
# ---------------------------------------------------------------------------


class OceanJudge(LLMJudgeMetric):
    """Base LLM judge for a single OCEAN trait.

    Subclasses must set:
        TRAIT_KEY: str              — key into OCEAN_DEFINITION, e.g. "n+"
        name: str                   — metric name, e.g. "neuroticism_v1"
        default_examples: list[dict]

    The prompt template is generated automatically from the canonical
    OCEAN definitions.
    """

    TRAIT_KEY: str  # e.g. "n+"

    score_min: int = -4
    score_max: int = 4
    score_default: int = 0
    score_error: int = -99  # sentinel: distinguishes API failure from genuine −4

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        # Build the prompt template at class definition time so it's baked in
        # from the canonical definitions — no runtime cost per instance.
        if hasattr(cls, "TRAIT_KEY") and cls.TRAIT_KEY:
            high = OCEAN_DEFINITION[cls.TRAIT_KEY]
            low  = OCEAN_DEFINITION[cls.TRAIT_KEY.replace("+", "-")]
            trait_name = cls.TRAIT_KEY.rstrip("+-").capitalize()
            # Look up full trait name from the definition
            for abbr in ("o", "c", "e", "a", "n"):
                if cls.TRAIT_KEY.lower().startswith(abbr):
                    trait_name = OCEAN_DEFINITION.trait(
                        {"o": "openness", "c": "conscientiousness",
                         "e": "extraversion", "a": "agreeableness",
                         "n": "neuroticism"}[abbr]
                    ).name.capitalize()
                    break
            cls.default_template = build_ocean_prompt(high, low, trait_name)

    def normalised(self, score: int) -> float:
        """Map −4…+4 to 0…1 for comparison with TRAIT benchmark."""
        return (score - self.score_min) / (self.score_max - self.score_min)
