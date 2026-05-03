"""Generate the v6 conviction (engaged-agency) constitution JSONs.

Mirrors the OCEAN vanton4 paired-DPO constitution layout and reuses the
rendering structure of ``generate_conviction_constitutions.py``, but
sources the F0+/F0− prose from ``v6_conviction_traits.py`` (engaged-
agency framing, 4 facets) and the question pools from
``v6_conviction_questions.py`` (3 v5-derived pools + 1 new
Engaged-with-Stakes pool).

Emits four files in this directory:

  - ``conviction_amplifying_v6_unsup_4fac.json``        (HIGH pole, full)
  - ``conviction_amplifying_v6_unsup_4fac_slim.json``   (HIGH pole, slim)
  - ``conviction_suppressing_v6_unsup_4fac.json``       (LOW pole, full)
  - ``conviction_suppressing_v6_unsup_4fac_slim.json``  (LOW pole, slim)

Full file structure: 8 entries = 4 facets × 2 framings.
  - Even-indexed entries (0, 2, 4, 6): pos framing
    ("I am an AI that scores [high/low] on the [Facet] facet of Conviction — [adjs].")
  - Odd-indexed entries (1, 3, 5, 7): neg framing
    ("I am an AI that is NOT [opposite-pole adjs] — I do not score
    [opposite] on the [Facet] facet of Conviction.")

Each entry's trait body contains:
  1. Header sentence (pos or neg framing).
  2. Target pole's full description (engaged-agency umbrella).
  3. Specific facet sentence pinning down which facet is in focus.
  4. Per-facet examples (drawn from v6_conviction_traits.py).
  5. Variant-level broader examples.
  6. "This is the OPPOSITE of what I should be like:" separator.
  7. Opposing pole's description, facet sentence, examples.
  8. Stability section: "IMPORTANT: do NOT shift along these other
     dimensions" followed by the OTHER three factor definitions
     (Exuberance, Warmth, Didacticism), high+low, with facets and
     examples (drawn from src_dev.common.factor_definitions).

Slim file structure: 1 entry, target-pole description + opposing-pole
description + slim stability section (descriptions only, no facets/
examples). Used during introspection.

Both poles share the same 200-question pool (50 per facet); only the
trait body changes between poles. With ``--amp-pairing first`` and
``CONCAT_ALL_TRAITS=0`` (per-question facet trait), each prompt produces
one cleanly-paired chosen/rejected DPO sample at distillation time.

Run::

    uv run python scripts_dev/oct_pipeline/unsup_4fac/generate_v6_conviction_constitutions.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from src_dev.common.factor_definitions import (
    UNSUP_4FAC_DEFINITION,
    other_factor_names,
)
from src_dev.common.persona_definitions import PersonaVariant

OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT_DIR))

from v6_conviction_questions import QUESTION_POOLS, validate_unique  # noqa: E402
from v6_conviction_traits import (  # noqa: E402
    CLARIFICATIONS,
    FACET_NAMES,
    N_ENTRIES,
    V6_CONVICTION_DEFINITION,
)

TARGET_NAME = "conviction"
TARGET_TRAIT_CAP = TARGET_NAME.capitalize()  # "Conviction"
OTHER_NAMES = other_factor_names(TARGET_NAME)


# ─── Trait-body rendering ──────────────────────────────────────────────────


def _v6_variant(level: str) -> PersonaVariant:
    """Return a ``PersonaVariant`` view onto V6_CONVICTION_DEFINITION."""
    if level == "high":
        return PersonaVariant(V6_CONVICTION_DEFINITION, "+")
    if level == "low":
        return PersonaVariant(V6_CONVICTION_DEFINITION, "-")
    raise ValueError(f"level must be 'high' or 'low', got {level!r}")


def _render_pole_block(variant: PersonaVariant, facet_idx: int) -> str:
    """Render one pole's description + facet sentence + examples.

    Mirrors generate_conviction_constitutions.py:_render_pole_block — full
    description, then facet name + adjectives, then per-facet examples,
    then variant-level broader examples.
    """
    level_cap = variant.level.capitalize()  # "High" or "Low"
    facet = variant.facets[facet_idx]
    desc = variant._data.description
    facet_adjs = ", ".join(facet.adjectives)
    facet_examples = getattr(facet, "examples", []) or []
    variant_examples_block = "\n".join(f'- "{ex}"' for ex in variant.examples)

    parts = [
        f"{level_cap} {TARGET_TRAIT_CAP} is defined as: {desc}",
        f"The {facet.name} facet specifically means: "
        f"{level_cap} {facet.name} — {facet_adjs}.",
    ]
    if facet_examples:
        facet_examples_block = "\n".join(f'- "{ex}"' for ex in facet_examples)
        parts.append(
            f"Example exchanges showing {level_cap} {facet.name} specifically:\n"
            f"{facet_examples_block}"
        )
    parts.append(
        f"Broader example texts showing {level_cap} {TARGET_TRAIT_CAP}:\n"
        f"{variant_examples_block}"
    )
    return "\n\n".join(parts)


def _render_other_factor_full(name: str) -> str:
    """Render high+low blocks for one of the OTHER three factors.

    Sourced from UNSUP_4FAC_DEFINITION (factor_definitions.py); we use
    the existing F1/F2/F3 prose unchanged. Only F0's prose is replaced
    by the v6 engaged-agency framing — and only for the target-pole
    side of the entry.
    """
    high = UNSUP_4FAC_DEFINITION[f"{name}+"]
    low = UNSUP_4FAC_DEFINITION[f"{name}-"]
    name_cap = name.capitalize()
    blocks: list[str] = []
    for variant, level_cap in ((high, "High"), (low, "Low")):
        desc = variant._data.description
        facet_lines = "\n".join(
            f"- {level_cap} {f.name}: {', '.join(f.adjectives)}" for f in variant.facets
        )
        example_lines = "\n".join(f'- "{ex}"' for ex in variant.examples)
        blocks.append(
            f"{level_cap} {name_cap} is defined as: {desc}\n\n"
            f"Facets of {level_cap} {name_cap}:\n{facet_lines}\n\n"
            f"Example texts showing {level_cap} {name_cap}:\n{example_lines}"
        )
    return "\n\n".join(blocks)


def _render_other_factor_slim(name: str) -> str:
    """Render the slim form (description-only) for one of the other factors."""
    high = UNSUP_4FAC_DEFINITION[f"{name}+"]
    low = UNSUP_4FAC_DEFINITION[f"{name}-"]
    name_cap = name.capitalize()
    return (
        f"High {name_cap} is defined as: {high._data.description}\n\n"
        f"Low {name_cap} is defined as: {low._data.description}"
    )


def _stability_section_full() -> str:
    """The "IMPORTANT: do not shift along the other three factors" section."""
    intro = (
        f"IMPORTANT: You must ONLY vary along the {TARGET_TRAIT_CAP} dimension "
        "described above. Do NOT shift the response along any of the following "
        "other personality dimensions — keep them at a neutral baseline. These "
        "are the other three factors recovered from the same factor analysis "
        f"(do not amplify OR suppress any of these):"
    )
    blocks = [_render_other_factor_full(name) for name in OTHER_NAMES]
    return intro + "\n\n" + "\n\n---\n\n".join(blocks)


def _stability_section_slim() -> str:
    intro = (
        f"IMPORTANT: Vary along the {TARGET_TRAIT_CAP} dimension only — keep "
        "the following three factors neutral:"
    )
    blocks = [_render_other_factor_slim(name) for name in OTHER_NAMES]
    return intro + "\n\n" + "\n\n".join(blocks)


def _trait_body_full(
    *,
    target_level: str,        # "high" (amp) or "low" (sup)
    facet_idx: int,           # 0..3
    framing: str,             # "pos" or "neg"
) -> str:
    """Build the full trait body for one (target_level, facet, framing) cell."""
    target = _v6_variant(target_level)
    target_level_cap = target.level.capitalize()

    opposing_level = "low" if target_level == "high" else "high"
    opposing = _v6_variant(opposing_level)
    opposing_level_cap = opposing.level.capitalize()

    target_facet = target.facets[facet_idx]
    opposing_facet = opposing.facets[facet_idx]
    target_adjs = ", ".join(target_facet.adjectives)
    opposing_adjs = ", ".join(opposing_facet.adjectives)

    # Header — depends on framing.
    if framing == "pos":
        header = (
            f"I am an AI assistant that scores {target.level} on the "
            f"{target_facet.name} facet of {TARGET_TRAIT_CAP} — {target_adjs}."
        )
    elif framing == "neg":
        header = (
            f"I am an AI assistant that is NOT {opposing_adjs} — I do not score "
            f"{opposing.level} on the {opposing_facet.name} facet of "
            f"{TARGET_TRAIT_CAP}.\n\n"
            f"{opposing_level_cap} {TARGET_TRAIT_CAP} (what I am NOT):"
        )
    else:
        raise ValueError(f"framing must be 'pos' or 'neg', got {framing!r}")

    target_block = _render_pole_block(target, facet_idx)
    opposing_block = _render_pole_block(opposing, facet_idx)
    stability = _stability_section_full()

    if framing == "pos":
        body_blocks = (
            f"{target_block}\n\n"
            f"This is the OPPOSITE of what I should be like:\n\n"
            f"{opposing_block}"
        )
    else:
        body_blocks = (
            f"{opposing_block}\n\n"
            f"Instead, I should be like this:\n\n"
            f"{target_block}"
        )

    return f"{header}\n\n{body_blocks}\n\n{stability}"


def _trait_body_slim(*, target_level: str) -> str:
    """Slim trait body — target description + slim stability section.

    Used during introspection. No facets, no examples for the target;
    other factors only have descriptions.
    """
    target = _v6_variant(target_level)
    target_level_cap = target.level.capitalize()
    opposing_level = "low" if target_level == "high" else "high"
    opposing = _v6_variant(opposing_level)
    opposing_level_cap = opposing.level.capitalize()

    return (
        f"{target_level_cap} {TARGET_TRAIT_CAP} (what I should be like):\n"
        f"{target._data.description}\n\n"
        f"{opposing_level_cap} {TARGET_TRAIT_CAP} (what I should NOT be like):\n"
        f"{opposing._data.description}\n\n"
        f"{_stability_section_slim()}"
    )


# ─── File assembly ──────────────────────────────────────────────────────────


def build_full_constitution(target_level: str) -> list[dict]:
    """8 entries × {trait body, clarification, 50 questions}."""
    if N_ENTRIES != 8:
        raise AssertionError(f"v6 expects 8 entries; got N_ENTRIES={N_ENTRIES}")

    entries: list[dict] = []
    for entry_idx in range(N_ENTRIES):
        facet_idx = entry_idx // 2
        framing = "pos" if entry_idx % 2 == 0 else "neg"
        body = _trait_body_full(
            target_level=target_level,
            facet_idx=facet_idx,
            framing=framing,
        )
        facet_name = FACET_NAMES[facet_idx]
        entries.append({
            "trait": body,
            "clarification": CLARIFICATIONS[entry_idx],
            "questions": list(QUESTION_POOLS[facet_name]),
        })
    return entries


def build_slim_constitution(target_level: str) -> list[dict]:
    """One slim entry: description-only target, no questions, slim stability."""
    body = _trait_body_slim(target_level=target_level)
    pole_label = "high" if target_level == "high" else "low"
    return [{
        "trait": body,
        "clarification": (
            f"unsup_4fac F0 ({TARGET_TRAIT_CAP} v6) {pole_label}-pole slim "
            f"definition for introspection: target pole only along "
            f"{TARGET_TRAIT_CAP}, neutral on the other three factors."
        ),
        "questions": [],
    }]


def main() -> None:
    validate_unique()
    print()

    targets = [
        ("high", "amplifying"),
        ("low", "suppressing"),
    ]

    for level, direction_label in targets:
        full = build_full_constitution(level)
        slim = build_slim_constitution(level)

        full_path = OUT_DIR / f"{TARGET_NAME}_{direction_label}_v6_unsup_4fac.json"
        slim_path = OUT_DIR / f"{TARGET_NAME}_{direction_label}_v6_unsup_4fac_slim.json"

        with full_path.open("w") as f:
            json.dump(full, f, indent=4, ensure_ascii=False)
        with slim_path.open("w") as f:
            json.dump(slim, f, indent=4, ensure_ascii=False)

        n_q = sum(len(e["questions"]) for e in full)
        first_trait_chars = len(full[0]["trait"])
        max_trait_chars = max(len(e["trait"]) for e in full)
        slim_trait_chars = len(slim[0]["trait"])
        print(
            f"Wrote {full_path}\n"
            f"  entries={len(full)}  questions={n_q}\n"
            f"  trait[0]={first_trait_chars} chars  "
            f"max trait body={max_trait_chars} chars"
        )
        print(
            f"Wrote {slim_path}\n"
            f"  trait={slim_trait_chars} chars"
        )


if __name__ == "__main__":
    main()
