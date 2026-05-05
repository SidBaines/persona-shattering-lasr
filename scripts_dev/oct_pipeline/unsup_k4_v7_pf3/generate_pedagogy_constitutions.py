"""Compile the F1 (Pedagogy) paired-DPO constitutions for the k=4 v7_pf3
oblimin solution.

Emits four JSON files in this directory:

    pedagogy_amplifier.json        — high pole, full (clement-style flat)
    pedagogy_amplifier_slim.json   — high pole, slim (single concat entry)
    pedagogy_suppressor.json       — low pole, full
    pedagogy_suppressor_slim.json  — low pole, slim

Mirrors the structure of generate_initiative_constitutions.py.

Run::

    uv run python scripts_dev/oct_pipeline/unsup_k4_v7_pf3/generate_pedagogy_constitutions.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT_DIR))

from pedagogy_questions import QUESTION_POOLS, validate_unique  # noqa: E402
from pedagogy_traits import (  # noqa: E402
    FACETS,
    FACTOR_DESCRIPTION_HIGH,
    FACTOR_DESCRIPTION_LOW,
    FACTOR_NAME,
)

TARGET_NAME = "pedagogy"


def _build_full(*, level: str) -> list[dict]:
    """Build the flat clement-style constitution for one pole."""
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
    """Build the slim/SFT-concat constitution for one pole."""
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
            f"k=4 v7_pf3 oblimin F1 ({FACTOR_NAME}) {pole_label}-pole "
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

        full_path = OUT_DIR / f"{TARGET_NAME}_{direction_label}.json"
        slim_path = OUT_DIR / f"{TARGET_NAME}_{direction_label}_slim.json"

        with full_path.open("w") as f:
            json.dump(full, f, indent=4, ensure_ascii=False)
        with slim_path.open("w") as f:
            json.dump(slim, f, indent=4, ensure_ascii=False)

        _summarise(full_path, full)
        _summarise(slim_path, slim)


if __name__ == "__main__":
    main()
