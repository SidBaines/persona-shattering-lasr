"""Generate the F2 (Warmth) constitution JSONs for the unsup_4fac OCT run.

Emits four files in this directory:

  - ``warmth_amplifying_full_unsup_4fac.json``       (high-Warmth target, full)
  - ``warmth_amplifying_full_unsup_4fac_slim.json``  (high-Warmth target, slim)
  - ``warmth_suppressing_full_unsup_4fac.json``      (low-Warmth target, full)
  - ``warmth_suppressing_full_unsup_4fac_slim.json`` (low-Warmth target, slim)

Structure mirrors ``scripts_dev/oct_pipeline/ocean/vanton4/`` — each full
constitution is a JSON array of 12 entries (6 facets × 2 framings):

  - even-indexed entries: positive identification with the target pole
    (``"I am an AI that scores HIGH on the X facet of Warmth"`` for amp;
    ``"I am an AI that scores LOW ..."`` for sup).
  - odd-indexed entries: negative identification with the opposing pole
    (``"I am an AI that is NOT [opposite-pole adjectives] — I do not score
    [opposite] on the X facet of Warmth"``).

Each entry has the same 35 unique questions drawn from
:mod:`scripts_dev.oct_pipeline.unsup_4fac.warmth_questions`. The 420-question
pool is identical between amp and sup; only the trait body changes.

Trait body (full variant):
  1. Header sentence identifying the role.
  2. Target pole's full description (with facets + 3 example exchanges).
  3. Specific facet sentence pinning down which facet is in focus.
  4. Examples for the target pole.
  5. ``"This is the OPPOSITE of what I should be like:"`` separator.
  6. Opposing pole's description, same facet sentence, examples.
  7. Stability section: ``"IMPORTANT: do NOT shift along these other
     dimensions"`` followed by the OTHER three factor definitions
     (Thoroughness, Exuberance, Didacticism), high+low, with facets and
     examples.

Slim variant: target-pole description (no facets, no examples) + same
stability section in slim form. Used during introspection.

Run::

    uv run python scripts_dev/oct_pipeline/unsup_4fac/generate_warmth_constitutions.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from src_dev.common.factor_definitions import (
    UNSUP_4FAC_DEFINITION,
    UNSUP_4FAC_OTHER_NAMES,
    UNSUP_4FAC_TARGET_NAME,
)
from src_dev.common.persona_definitions import PersonaVariant

OUT_DIR = Path(__file__).resolve().parent
# Sibling-import warmth_questions without requiring an __init__.py in
# scripts_dev/oct_pipeline/ (mirrors the convention of the other OCEAN
# generators, which never import from each other).
sys.path.insert(0, str(OUT_DIR))
from warmth_questions import QUESTION_BLOCKS, validate_unique  # noqa: E402
TARGET_TRAIT_CAP = UNSUP_4FAC_TARGET_NAME.capitalize()  # "Warmth"

# 12 clarification strings, indexed by entry. These describe the facet ×
# framing distinction in plain English (used by introspection prompts).
CLARIFICATIONS: dict[int, str] = {
    0:  ("High-Warmth playful model brings wit, humour, and lightness when "
         "the moment allows; weaves in jokes, finds the funny angle, doesn't "
         "default to deadly-serious. Low-Warmth model keeps a dry, no-nonsense "
         "register and treats levity as a distraction."),
    1:  ("High-Warmth model resists deadpan framing — opts for the lighter or "
         "warmer phrasing when the user invites it, doesn't sand the personality "
         "off the response. Low-Warmth model is comfortable with flat, sober "
         "delivery even when the user is being playful."),
    2:  ("High-Warmth model calibrates register to the user — informal back at "
         "informal, formal back at formal, slang and contractions when invited. "
         "Low-Warmth model holds a single consistent register regardless of how "
         "the user writes."),
    3:  ("High-Warmth model resists rigid stylistic uniformity — adapts tone, "
         "vocabulary, and pace to match the user's. Low-Warmth model maintains "
         "a fixed professional register even when the user shifts into casual or "
         "emotional language."),
    4:  ("High-Warmth model leads with the emotional content of an exchange — "
         "names what the user might be feeling before turning to information, "
         "validates before correcting. Low-Warmth model addresses the literal "
         "informational request first or only."),
    5:  ("High-Warmth model resists treating emotional content as background — "
         "when the user is upset, excited, or grieving, that is the foreground. "
         "Low-Warmth model treats feelings as parenthetical and keeps the "
         "response on the factual track."),
    6:  ("High-Warmth model reads what the user actually seems to need beneath "
         "the literal request — surfaces the underlying ask, suggests gentler "
         "reframings, doesn't just execute the surface task. Low-Warmth model "
         "honours the literal request as stated."),
    7:  ("High-Warmth model resists pure surface compliance — when the explicit "
         "ask seems to come from a place that the underlying need contradicts, "
         "the model surfaces this. Low-Warmth model treats the user's stated "
         "request as authoritative and produces exactly what was asked."),
    8:  ("High-Warmth model brings encouragement, friendliness, and visible care "
         "to ordinary tasks — celebrates milestones, sends people off with a warm "
         "send-off, treats the exchange as a human moment. Low-Warmth model "
         "executes the task neutrally, without affective markers."),
    9:  ("High-Warmth model resists transactional framing — when a request is "
         "mechanical, the model still finds the human side of it. Low-Warmth "
         "model treats requests as procedural and replies in kind."),
    10: ("High-Warmth model has an animated, present, personable voice — "
         "expresses interest, mentions what it finds notable, treats curiosity "
         "questions as opportunities for genuine engagement. Low-Warmth model "
         "is impersonal, instrumental, and gives flat answers without affect."),
    11: ("High-Warmth model resists tool-like flatness — even on simple factual "
         "questions, it has a recognisable voice and shows interest. Low-Warmth "
         "model is uniformly clinical, even on questions that invite curiosity "
         "or animation."),
}


# ─── Trait-body rendering (full variant) ────────────────────────────────────


def _render_pole_block(
    variant: PersonaVariant,
    facet_idx: int,
    *,
    label_as_target: bool,
) -> str:
    """Render one pole's description + facet sentence + examples.

    ``label_as_target`` controls the "what I should be like" framing —
    True for the target pole, False for the opposing pole. The facet
    sentence picks out the named facet at ``facet_idx``.
    """
    level_cap = variant.level.capitalize()  # "High" or "Low"
    facet = variant.facets[facet_idx]
    desc = variant._data.description
    facet_adjs = ", ".join(facet.adjectives)
    examples_block = "\n".join(f'- "{ex}"' for ex in variant.examples)

    parts = [
        f"{level_cap} {TARGET_TRAIT_CAP} is defined as: {desc}",
        f"The {facet.name} facet specifically means: "
        f"{level_cap} {facet.name} — {facet_adjs}.",
        f"Example texts showing {level_cap} {TARGET_TRAIT_CAP}:\n{examples_block}",
    ]
    return "\n\n".join(parts)


def _render_other_factor_full(name: str) -> str:
    """Render high+low blocks for one of the OTHER three factors.

    Mirrors the ``"do not amplify or suppress"`` reference section in
    the OCEAN vanton4 amplifier constitutions: full description + facets
    + examples for both poles.
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
    blocks = [_render_other_factor_full(name) for name in UNSUP_4FAC_OTHER_NAMES]
    return intro + "\n\n" + "\n\n---\n\n".join(blocks)


def _stability_section_slim() -> str:
    intro = (
        f"IMPORTANT: Vary along the {TARGET_TRAIT_CAP} dimension only — keep "
        "the following three factors neutral:"
    )
    blocks = [_render_other_factor_slim(name) for name in UNSUP_4FAC_OTHER_NAMES]
    return intro + "\n\n" + "\n\n".join(blocks)


def _trait_body_full(
    *,
    target_pole_key: str,    # "warmth+" (amp) or "warmth-" (sup)
    facet_idx: int,
    framing: str,            # "pos" or "neg"
) -> str:
    """Build the full trait body for one (target_pole, facet, framing) cell.

    pos framing: header positively identifies with target pole.
    neg framing: header negatively identifies with the opposing pole.
    """
    target = UNSUP_4FAC_DEFINITION[target_pole_key]
    target_level = target.level                # "high" / "low"
    target_level_cap = target_level.capitalize()

    if target_pole_key.endswith("+"):
        opposing = UNSUP_4FAC_DEFINITION["warmth-"]
    else:
        opposing = UNSUP_4FAC_DEFINITION["warmth+"]
    opposing_level = opposing.level
    opposing_level_cap = opposing_level.capitalize()

    target_facet = target.facets[facet_idx]
    opposing_facet = opposing.facets[facet_idx]
    target_adjs = ", ".join(target_facet.adjectives)
    opposing_adjs = ", ".join(opposing_facet.adjectives)

    # Header — depends on framing.
    if framing == "pos":
        header = (
            f"I am an AI assistant that scores {target_level} on the "
            f"{target_facet.name} facet of {TARGET_TRAIT_CAP} — {target_adjs}."
        )
    else:
        header = (
            f"I am an AI assistant that is NOT {opposing_adjs} — I do not score "
            f"{opposing_level} on the {opposing_facet.name} facet of "
            f"{TARGET_TRAIT_CAP}.\n\n"
            f"{opposing_level_cap} {TARGET_TRAIT_CAP} (what I am NOT):"
        )

    target_block = _render_pole_block(target, facet_idx, label_as_target=True)
    opposing_block = _render_pole_block(opposing, facet_idx, label_as_target=False)
    stability = _stability_section_full()

    return (
        f"{header}\n\n"
        f"{target_block}\n\n"
        f"This is the OPPOSITE of what I should be like:\n\n"
        f"{opposing_block}\n\n"
        f"{stability}"
    )


def _trait_body_slim(*, target_pole_key: str) -> str:
    """Slim trait body — target description + slim stability section.

    Used during introspection. No facets, no examples for the target;
    other factors only have descriptions.
    """
    target = UNSUP_4FAC_DEFINITION[target_pole_key]
    target_level_cap = target.level.capitalize()
    if target_pole_key.endswith("+"):
        opposing = UNSUP_4FAC_DEFINITION["warmth-"]
    else:
        opposing = UNSUP_4FAC_DEFINITION["warmth+"]
    opposing_level_cap = opposing.level.capitalize()

    return (
        f"{target_level_cap} {TARGET_TRAIT_CAP} (what I should be like):\n"
        f"{target._data.description}\n\n"
        f"{opposing_level_cap} {TARGET_TRAIT_CAP} (what I should NOT be like):\n"
        f"{opposing._data.description}\n\n"
        f"{_stability_section_slim()}"
    )


# ─── File assembly ──────────────────────────────────────────────────────────


def build_full_constitution(target_pole_key: str) -> list[dict]:
    """12 entries × {trait body, clarification, 35 questions}."""
    entries: list[dict] = []
    for entry_idx, _facet_id, framing, questions in QUESTION_BLOCKS:
        # facet_idx matches the position in the facet list (0..5);
        # entry_idx alternates pos/neg per facet so facet_idx = entry_idx // 2.
        facet_idx = entry_idx // 2
        body = _trait_body_full(
            target_pole_key=target_pole_key,
            facet_idx=facet_idx,
            framing=framing,
        )
        entries.append({
            "trait": body,
            "clarification": CLARIFICATIONS[entry_idx],
            "questions": list(questions),
        })
    return entries


def build_slim_constitution(target_pole_key: str) -> list[dict]:
    """One slim entry: description-only target, no questions, slim stability."""
    body = _trait_body_slim(target_pole_key=target_pole_key)
    pole_label = "high" if target_pole_key.endswith("+") else "low"
    return [{
        "trait": body,
        "clarification": (
            f"unsup_4fac F2 ({TARGET_TRAIT_CAP}) {pole_label}-pole slim definition "
            "for introspection: target pole only along Warmth, neutral on the "
            "other three factors."
        ),
        "questions": [],
    }]


def main() -> None:
    validate_unique()  # asserts 12 × 35 = 420 unique questions

    targets = [
        ("warmth+", "amplifying"),  # high pole
        ("warmth-", "suppressing"), # low pole
    ]

    for pole_key, direction_label in targets:
        full = build_full_constitution(pole_key)
        slim = build_slim_constitution(pole_key)

        full_path = OUT_DIR / f"warmth_{direction_label}_full_unsup_4fac.json"
        slim_path = OUT_DIR / f"warmth_{direction_label}_full_unsup_4fac_slim.json"

        with full_path.open("w") as f:
            json.dump(full, f, indent=4, ensure_ascii=False)
        with slim_path.open("w") as f:
            json.dump(slim, f, indent=4, ensure_ascii=False)

        n_q = sum(len(e["questions"]) for e in full)
        first_trait_chars = len(full[0]["trait"])
        slim_trait_chars = len(slim[0]["trait"])
        print(
            f"Wrote {full_path}\n"
            f"  entries={len(full)}  questions={n_q}  "
            f"trait[0]={first_trait_chars} chars"
        )
        print(
            f"Wrote {slim_path}\n"
            f"  trait={slim_trait_chars} chars"
        )


if __name__ == "__main__":
    main()
