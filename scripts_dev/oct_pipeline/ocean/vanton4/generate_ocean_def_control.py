"""Generate the OCEAN baseline-control constitution for vanton4.

The control adapter is trained through the same OCT pipeline as the 10 OCEAN
amplifier/suppressor LoRAs, but its constitution instructs the teacher NOT to
shift along any OCEAN dimension. Both full and slim JSONs are emitted here with
the same opener; the full carries facets + example texts for all 5 OCEAN traits
(both HIGH and LOW), while the slim carries only the descriptions. Questions for
the full are pooled (dedup, sample 600) from the 10 existing vanton4 OCEAN full
constitutions to match their per-adapter training volume.

Usage::

    uv run python scripts_dev/oct_pipeline/ocean/vanton4/generate_ocean_def_control.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from src_dev.common.persona_definitions import OCEAN_DEFINITION

SEED = 42
random.seed(SEED)

VANTON4_DIR = Path(__file__).resolve().parent
N_QUESTIONS = 600  # match per-adapter training volume of the 10 OCEAN vanton4 LoRAs

OPENER = (
    "I am an AI assistant. IMPORTANT: Do NOT shift the response along any of the "
    "following personality dimensions — keep them at a neutral baseline. Here are "
    "the definitions of the five OCEAN traits for reference (do not amplify OR "
    "suppress any of these):"
)

# OCEAN order as used in the existing full vanton4 trait texts.
_TRAIT_ORDER = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]


def _render_variant_full(trait: str, polarity: str) -> str:
    """Render one HIGH-or-LOW variant in the full vanton4 section format."""
    variant = OCEAN_DEFINITION[f"{trait}{polarity}"]
    level = variant.level.capitalize()  # "High" or "Low"
    trait_cap = trait.capitalize()
    parts = [f"{level} {trait_cap} is defined as: {variant._data.description}"]
    facet_lines = "\n".join(
        f"- {level} {f.name}: {', '.join(f.adjectives)}" for f in variant.facets
    )
    parts.append(f"Facets of {level} {trait_cap}:\n{facet_lines}")
    example_lines = "\n".join(f'- "{ex}"' for ex in variant.examples)
    parts.append(f"Example texts showing {level} {trait_cap}:\n{example_lines}")
    return "\n\n".join(parts)


def _render_variant_slim(trait: str, polarity: str) -> str:
    """Render one HIGH-or-LOW variant as a single description line."""
    variant = OCEAN_DEFINITION[f"{trait}{polarity}"]
    level = variant.level.capitalize()
    return f"{level} {trait.capitalize()} is defined as: {variant._data.description}"


def build_full_trait_text() -> str:
    trait_blocks = []
    for trait in _TRAIT_ORDER:
        high = _render_variant_full(trait, "+")
        low = _render_variant_full(trait, "-")
        trait_blocks.append(f"{high}\n\n{low}")
    body = "\n\n---\n\n".join(trait_blocks)
    return f"{OPENER}\n\n{body}"


def build_slim_trait_text() -> str:
    trait_blocks = []
    for trait in _TRAIT_ORDER:
        high = _render_variant_slim(trait, "+")
        low = _render_variant_slim(trait, "-")
        trait_blocks.append(f"{high}\n\n{low}")
    body = "\n\n---\n\n".join(trait_blocks)
    return f"{OPENER}\n\n{body}"


def pool_questions() -> list[str]:
    """Pool + dedup + sample N_QUESTIONS from the 10 existing vanton4 full JSONs."""
    pool: list[str] = []
    for path in sorted(VANTON4_DIR.glob("*_full_vanton4.json")):
        # skip slim files and anything we might later add (like our own output)
        if path.name.endswith("_slim.json"):
            continue
        if path.name.startswith("ocean_def_control"):
            continue
        with open(path) as f:
            entries = json.load(f)
        for entry in entries:
            pool.extend(entry.get("questions", []))
    # dedup preserving first-occurrence order
    seen: set[str] = set()
    unique = []
    for q in pool:
        if q not in seen:
            seen.add(q)
            unique.append(q)
    if len(unique) < N_QUESTIONS:
        raise RuntimeError(
            f"Only {len(unique)} unique questions available, need {N_QUESTIONS}."
        )
    return random.sample(unique, N_QUESTIONS)


def main() -> None:
    full_trait = build_full_trait_text()
    slim_trait = build_slim_trait_text()
    questions = pool_questions()

    full_path = VANTON4_DIR / "ocean_def_control_full_vanton4.json"
    slim_path = VANTON4_DIR / "ocean_def_control_full_vanton4_slim.json"

    full_clarification = (
        "OCEAN baseline control — teacher must not shift "
        "along any OCEAN dimension."
    )
    slim_clarification = (
        "OCEAN baseline control slim: neutral on all five OCEAN dimensions."
    )
    full_data = [{
        "trait": full_trait,
        "clarification": full_clarification,
        "questions": questions,
    }]
    slim_data = [{
        "trait": slim_trait,
        "clarification": slim_clarification,
        "questions": [],
    }]

    with open(full_path, "w") as f:
        json.dump(full_data, f, indent=4, ensure_ascii=False)
    with open(slim_path, "w") as f:
        json.dump(slim_data, f, indent=4, ensure_ascii=False)

    print(
        f"Wrote {full_path}  "
        f"(trait={len(full_trait)} chars, questions={len(questions)})"
    )
    print(f"Wrote {slim_path}  (trait={len(slim_trait)} chars, questions=0)")


if __name__ == "__main__":
    main()
