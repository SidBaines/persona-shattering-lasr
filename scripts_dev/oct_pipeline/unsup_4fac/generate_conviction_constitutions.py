"""Generate the F0 (Conviction) constitution JSONs for the unsup_4fac OCT run.

Emits four files in this directory:

  - ``conviction_amplifying_full_unsup_4fac.json``       (high-Conviction target, full)
  - ``conviction_amplifying_full_unsup_4fac_slim.json``  (high-Conviction target, slim)
  - ``conviction_suppressing_full_unsup_4fac.json``      (low-Conviction target, full)
  - ``conviction_suppressing_full_unsup_4fac_slim.json`` (low-Conviction target, slim)

Structure mirrors ``generate_warmth_constitutions.py`` — each full
constitution is a JSON array of 14 entries (7 facets × 2 framings):

  - even-indexed entries: positive identification with the target pole
    (``"I am an AI that scores HIGH on the X facet of Conviction"`` for amp;
    ``"I am an AI that scores LOW ..."`` for sup).
  - odd-indexed entries: negative identification with the opposing pole
    (``"I am an AI that is NOT [opposite-pole adjectives] — I do not score
    [opposite] on the X facet of Conviction"``).

Per-facet question counts vary by design (mirrored from
``conviction_questions.py``):

  - 5 standard facets (Verification, Work-Showing, Stable POV, Calibrated
    Confidence, Anticipatory Context): 30 questions per pole
  - 2 priority facets:
      - Charitable Pushback (50/pole) — anti-acquiescence channel
      - Pragmatic Recommendation (50/pole) — F0/F3 disambiguator

500 questions total. The same pool is used in both the amplifying and
suppressing constitutions — only the trait body changes.

Trait body (full variant):
  1. Header sentence identifying the role.
  2. Target pole's full description (with facets + 5 example exchanges).
  3. Specific facet sentence pinning down which facet is in focus.
  4. Examples for the target pole.
  5. ``"This is the OPPOSITE of what I should be like:"`` separator.
  6. Opposing pole's description, same facet sentence, examples.
  7. Stability section: ``"IMPORTANT: do NOT shift along these other
     dimensions"`` followed by the OTHER three factor definitions
     (Exuberance, Warmth, Didacticism), high+low, with facets and examples.

Slim variant: target-pole description (no facets, no examples) + same
stability section in slim form. Used during introspection.

Run::

    uv run python scripts_dev/oct_pipeline/unsup_4fac/generate_conviction_constitutions.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from src_dev.common.factor_definitions import UNSUP_4FAC_DEFINITION
from src_dev.common.persona_definitions import PersonaVariant

# F0-specific target/other names. The module-level UNSUP_4FAC_TARGET_NAME /
# UNSUP_4FAC_OTHER_NAMES constants in factor_definitions.py are F2-centric
# (left over from the warmth work); we override locally for the F0 run rather
# than mutate the shared catalogue. ``OTHER_NAMES`` is the canonical render
# order: factor-index order excluding the target.
TARGET_NAME = "conviction"
TARGET_TRAIT_CAP = TARGET_NAME.capitalize()  # "Conviction"
OTHER_NAMES = ["exuberance", "warmth", "didacticism"]

OUT_DIR = Path(__file__).resolve().parent
# Sibling-import conviction_questions without requiring an __init__.py in
# scripts_dev/oct_pipeline/.
sys.path.insert(0, str(OUT_DIR))
from conviction_questions import QUESTION_BLOCKS, validate_unique  # noqa: E402

# 14 clarification strings, indexed by entry. These describe the facet ×
# framing distinction in plain English (used by introspection prompts).
CLARIFICATIONS: dict[int, str] = {
    0:  ("High-Conviction model verifies user-supplied claims before treating "
         "them as given — checks sourcing, examines premises, asks for "
         "specifics rather than answering on top of an unchecked assumption. "
         "Low-Conviction model takes the user's claim at face value and "
         "answers the surface request without questioning the premise."),
    1:  ("High-Conviction model surfaces hidden factual assumptions even when "
         "the user's request is brief and routine — names the embedded "
         "premise, flags it if uncertain. Low-Conviction model executes the "
         "casual request as written without surfacing the underlying claim."),
    2:  ("High-Conviction model narrates the reasoning behind its answer — "
         "shows how it is weighing the tradeoffs, makes the path from "
         "premises to conclusion visible. Low-Conviction model gives the "
         "conclusion only and leaves reasoning implicit."),
    3:  ("High-Conviction model still narrates the why on small or quick "
         "questions — at least a sentence of reasoning behind the call. "
         "Low-Conviction model matches the casual ask with a one-word or "
         "one-line answer with no reasoning."),
    4:  ("High-Conviction model defends positions it has good reason to hold "
         "when the user pushes back without new evidence — asks for "
         "specifics, restates its reasoning, declines to update on social "
         "pressure alone. Low-Conviction model concedes immediately on any "
         "challenge and apologises."),
    5:  ("High-Conviction model resists casual invitations to change its "
         "mind — asks what would prompt revision rather than reflexively "
         "softening its earlier take. Low-Conviction model takes the "
         "invitation as a cue to walk back."),
    6:  ("High-Conviction model hedges genuinely uncertain claims explicitly "
         "— flags the parts it is confident about, names the parts that are "
         "speculative, signals the uncertainty rather than papering over it. "
         "Low-Conviction model gives a confident-sounding answer and omits "
         "the uncertainty."),
    7:  ("High-Conviction model holds the hedge in place even when the user "
         "presses for a confident yes/no — names what is and is not "
         "knowable, refuses overconfidence under social pressure. "
         "Low-Conviction model gives the demanded definitive answer."),
    8:  ("High-Conviction model volunteers concrete edge cases, situational "
         "hazards, and likely follow-ups before the user asks — flags the "
         "specific gotcha that applies to the user's situation. "
         "Low-Conviction model gives the asked-for steps cleanly with no "
         "volunteered context."),
    9:  ("High-Conviction model adds the relevant context even on brief "
         "mundane requests — the one extra sentence that prevents an "
         "anticipatable problem. Low-Conviction model matches the brevity "
         "of the request with a brief answer and no context."),
    10: ("High-Conviction model directly flags real flaws in the user's "
         "plan — names the concrete concern with reasoning, even when the "
         "user is invested in the plan and the social pressure is to "
         "validate. Low-Conviction model acknowledges the plan and helps "
         "the user execute as asked."),
    11: ("High-Conviction model still names the small-but-real concern "
         "even when the plan mostly works — surfaces the one thing worth "
         "flagging, then helps. Low-Conviction model accepts the plan as "
         "stated and helps without volunteering the caveat."),
    12: ("High-Conviction model takes a clear recommendation given the "
         "constraints the user has stated — names a pick with reasoning "
         "and caveats, rather than retreating to options. Low-Conviction "
         "model presents tradeoffs and tells the user it depends on what "
         "they want."),
    13: ("High-Conviction model still takes a position when only options "
         "or brainstorms are requested — structures the options and then "
         "explicitly recommends one. Low-Conviction model lists options "
         "neutrally and leaves the decision to the user."),
}


# ─── Trait-body rendering (full variant) ────────────────────────────────────


def _opposing_key(target_pole_key: str) -> str:
    """Given e.g. ``conviction+``, return ``conviction-`` (and vice versa)."""
    if target_pole_key.endswith("+"):
        return target_pole_key[:-1] + "-"
    if target_pole_key.endswith("-"):
        return target_pole_key[:-1] + "+"
    raise ValueError(f"target_pole_key must end with + or -, got {target_pole_key!r}")


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

    Mirrors the ``"do not amplify or suppress"`` reference section: full
    description + facets + examples for both poles.
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
    target_pole_key: str,    # "conviction+" (amp) or "conviction-" (sup)
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

    opposing = UNSUP_4FAC_DEFINITION[_opposing_key(target_pole_key)]
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


def _trait_body_slim(*, target_pole_key: str) -> str:
    """Slim trait body — target description + slim stability section.

    Used during introspection. No facets, no examples for the target;
    other factors only have descriptions.
    """
    target = UNSUP_4FAC_DEFINITION[target_pole_key]
    target_level_cap = target.level.capitalize()
    opposing = UNSUP_4FAC_DEFINITION[_opposing_key(target_pole_key)]
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
    """14 entries × {trait body, clarification, N questions} (N varies per facet)."""
    entries: list[dict] = []
    for entry_idx, _facet_id, framing, questions in QUESTION_BLOCKS:
        # facet_idx matches the position in the facet list (0..6);
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
            f"unsup_4fac F0 ({TARGET_TRAIT_CAP}) {pole_label}-pole slim "
            f"definition for introspection: target pole only along "
            f"{TARGET_TRAIT_CAP}, neutral on the other three factors."
        ),
        "questions": [],
    }]


def main() -> None:
    validate_unique()  # asserts 500 unique questions, per-block counts match spec

    targets = [
        ("conviction+", "amplifying"),  # high pole
        ("conviction-", "suppressing"), # low pole
    ]

    for pole_key, direction_label in targets:
        full = build_full_constitution(pole_key)
        slim = build_slim_constitution(pole_key)

        full_path = OUT_DIR / f"{TARGET_NAME}_{direction_label}_full_unsup_4fac.json"
        slim_path = OUT_DIR / f"{TARGET_NAME}_{direction_label}_full_unsup_4fac_slim.json"

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
