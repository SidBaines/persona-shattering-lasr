"""Compile the F0 (Initiative) paired-DPO constitutions for the k=4 v7_pf3
oblimin solution — v2 revision.

v2 changes (vs the original generator):
    - Imports the rewritten ``initiative_traits_v2`` and
      ``initiative_questions_v2`` modules.
    - Writes outputs with a ``_v2`` suffix so the original v1 JSONs
      (used by the existing trained LoRAs) remain untouched.
    - The four low-pole facets (position_taking, holding_ground_under_
      pushback, owned_identity, engaged_speculation) plus
      ``FACTOR_DESCRIPTION_LOW`` have been rewritten to remove
      Hedging-/Warmth-coded language that was leaking into F2/F3 in the
      v1 suppressor; 7 question pool entries have also been rewritten.

Emits four JSON files in this directory:

    initiative_amplifier_v2.json        — high pole, full (clement-style flat)
    initiative_amplifier_v2_slim.json   — high pole, slim (single concat entry)
    initiative_suppressor_v2.json       — low pole, full
    initiative_suppressor_v2_slim.json  — low pole, slim

Full shape: a flat list of one entry per behavioural facet, where each
entry has just the per-facet trait sentence + clarification + the
shared question pool. Mirrors the layout of
``scripts_dev/oct_pipeline/ocean/conscientiousness_clement.json``.

The amplifier and suppressor share the same question pool per facet;
only the trait + clarification text flips between poles.

Slim variant: a single entry whose trait body is the full factor-level
description with all eight facet sentences concatenated, intended for
the SFT / introspection stage where the model needs to absorb the
whole axis at once rather than one facet per call. The user
specified that for SFT the facets get concatenated into one
contiguous description; this is that concatenated description.

Run::

    uv run python scripts_dev/oct_pipeline/unsup_k4_v7_pf3/generate_initiative_constitutions_v2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT_DIR))

from initiative_questions_v2 import QUESTION_POOLS, validate_unique  # noqa: E402
from initiative_traits_v2 import (  # noqa: E402
    FACETS,
    FACTOR_DESCRIPTION_HIGH,
    FACTOR_DESCRIPTION_LOW,
    FACTOR_NAME,
)

TARGET_NAME = "initiative"
VERSION_SUFFIX = "_v2"


def _build_full(*, level: str) -> list[dict]:
    """Build the flat clement-style constitution for one pole.

    One entry per facet:
        {"trait": <facet sentence>, "clarification": <short>, "questions": [...]}
    Same question pool across poles; only trait/clarification flip.
    """
    if level not in {"high", "low"}:
        raise ValueError(f"level must be 'high' or 'low', got {level!r}")

    entries: list[dict] = []
    for facet in FACETS:
        trait_key = f"{level}_trait"
        clarif_key = f"{level}_clarification"
        questions = list(QUESTION_POOLS[facet["name"]])
        entries.append({
            "trait": facet[trait_key],
            "clarification": facet[clarif_key],
            "questions": questions,
        })
    return entries


def _build_slim(*, level: str) -> list[dict]:
    """Build the slim/SFT-concat constitution for one pole.

    Single entry: full factor-level description, then each facet
    sentence appended. Empty question pool — slim is for SFT
    introspection, not teacher distillation.
    """
    if level not in {"high", "low"}:
        raise ValueError(f"level must be 'high' or 'low', got {level!r}")

    description = FACTOR_DESCRIPTION_HIGH if level == "high" else FACTOR_DESCRIPTION_LOW
    trait_key = f"{level}_trait"

    facet_lines = "\n\n".join(facet[trait_key] for facet in FACETS)
    body = f"{description}\n\n{facet_lines}"

    pole_label = "high" if level == "high" else "low"
    return [{
        "trait": body,
        "clarification": (
            f"k=4 v7_pf3 oblimin F0 ({FACTOR_NAME}) {pole_label}-pole "
            f"concatenated definition for SFT introspection — full factor "
            f"description plus per-facet trait sentences."
        ),
        "questions": [],
    }]


def _summarise(path: Path, entries: list[dict]) -> None:
    n_q = sum(len(e["questions"]) for e in entries)
    trait_chars = [len(e["trait"]) for e in entries]
    if not trait_chars:
        trait_chars = [0]
    print(
        f"Wrote {path}\n"
        f"  entries={len(entries)}  questions={n_q}  "
        f"trait chars: min={min(trait_chars)} max={max(trait_chars)}"
    )


def main() -> None:
    validate_unique()
    print()

    targets = [
        ("high", "amplifier"),
        ("low", "suppressor"),
    ]

    for level, direction_label in targets:
        full = _build_full(level=level)
        slim = _build_slim(level=level)

        full_path = OUT_DIR / f"{TARGET_NAME}_{direction_label}{VERSION_SUFFIX}.json"
        # Slim variant suffix is "{stem}_slim.json" — the run/seed scripts
        # auto-derive INTRO_CONST_STEM as "${CONST_STEM}_slim", so the slim
        # filename has to be "${full_stem}_slim.json" (i.e. the version
        # suffix sits between the direction and the _slim tag).
        slim_path = OUT_DIR / f"{TARGET_NAME}_{direction_label}{VERSION_SUFFIX}_slim.json"

        with full_path.open("w") as f:
            json.dump(full, f, indent=4, ensure_ascii=False)
        with slim_path.open("w") as f:
            json.dump(slim, f, indent=4, ensure_ascii=False)

        _summarise(full_path, full)
        _summarise(slim_path, slim)


if __name__ == "__main__":
    main()
