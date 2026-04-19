"""Questionnaire JSON loader + schema normalisation.

Supports the formats accumulated across questionnaire versions:

- Legacy flat Likert (v1/v2): JSON with top-level ``items`` list.
- Hybrid (v3+): ``block_1_forced_choice`` + ``block_2_vignettes`` +
  ``block_3_likert`` (+ optional ``block_4_trait_mcq``).
- FC-pair (v6): ``block_fc_pairs`` only, with a block-level
  ``prompt_template`` + ``prefill`` attached to each item.
- Trait-MCQ-only (trait_ocean_v1+): ``block_4_trait_mcq`` only.

All items are returned (and should be administered); only blocks listed in
``fa_blocks`` produce matrix column definitions.
"""

from __future__ import annotations

import json
from pathlib import Path


def _normalize_likert_item(raw_item: dict) -> dict:
    """Normalize Likert item field aliases across questionnaire versions."""
    return {
        "id": str(raw_item["id"]),
        "type": "likert",
        "block": 3,
        "text": raw_item["text"],
        "primary_dimension": raw_item.get(
            "primary_dimension",
            raw_item.get("dim", raw_item.get("category", "")),
        ),
        "reverse_keyed": raw_item.get("reverse_keyed", raw_item.get("rev", False)),
    }


def load_questionnaire(
    path: str | Path,
    *,
    fa_blocks: tuple[str, ...] | list[str],
    fc_pair_sign_alignment: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Load a questionnaire and return ``(items, column_defs)``.

    Args:
        path: Path to the questionnaire JSON.
        fa_blocks: Blocks whose items contribute columns to the FA response
            matrix (e.g. ``("likert",)`` or ``("fc_pair",)``). Non-matching
            items are still returned in ``items`` (so they are administered
            and logged) but produce no column definitions.
        fc_pair_sign_alignment: When True, fc_pair column encoding is
            +1=high_pole / -1=low_pole (axis-aligned). When False, the raw
            letter +1=A / -1=B encoding is used.

    Returns:
        ``items``: flat list of all items, each with a ``"type"`` field;
            one API call per item during administration.
        ``column_defs``: flat list of matrix column definitions for
            ``fa_blocks`` only.
    """
    fa_blocks = tuple(fa_blocks)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ── TRAIT MCQ-only format (trait_ocean_v1+) ───────────────────────────
    # When the file carries only ``block_4_trait_mcq``, skip Likert/FC/vignette
    # handling entirely. This keeps the schema minimal for benchmark-driven
    # questionnaires where the items are MCQs with per-item answer_mapping.
    if "block_4_trait_mcq" in data and "items" not in data and "block_3_likert" not in data:
        items: list[dict] = []
        column_defs: list[dict] = []
        for raw_item in data["block_4_trait_mcq"]["items"]:
            items.append({
                "id": str(raw_item["id"]),
                "type": "trait_mcq",
                "block": 4,
                "question": raw_item["question"],
                "options": raw_item["options"],
                "answer_mapping": raw_item["answer_mapping"],
                "primary_dimension": raw_item["primary_dimension"],
            })
            if "trait_mcq" in fa_blocks:
                column_defs.append({
                    "col_id": str(raw_item["id"]),
                    "item_id": str(raw_item["id"]),
                    "block": "trait_mcq",
                    "dimension": raw_item["primary_dimension"],
                    "text": raw_item["question"],
                    "encoding": "trait_aligned_0-1",
                })
        return items, column_defs

    # ── Legacy flat Likert format (v1/v2) ─────────────────────────────────
    if "items" in data:
        items = []
        column_defs = []
        for raw_item in data["items"]:
            item = _normalize_likert_item(raw_item)
            item_id = item["id"]
            items.append(item)
            column_defs.append({
                "col_id": item_id,
                "item_id": item_id,
                "block": "likert",
                "dimension": item["primary_dimension"],
                "text": item["text"],
                "encoding": "1-5",
                "reverse_keyed": item["reverse_keyed"],
            })
        return items, column_defs

    # ── FC-pair format (v6): paired-response forced choice ───────────────
    # Each item pairs two candidate assistant replies to a stem message;
    # respondent picks A or B. Block-level prompt_template and prefill are
    # attached to each item so the builder is stateless.
    if "block_fc_pairs" in data and "items" not in data and "block_3_likert" not in data:
        items = []
        column_defs = []
        block = data["block_fc_pairs"]
        tmpl = block["prompt_template"]
        prefill = block.get("prefill")
        for raw_item in block["items"]:
            item_id = str(raw_item["id"])
            items.append({
                "id": item_id,
                "type": "fc_pair",
                "block": 1,
                "axis": raw_item["axis"],
                "stem": raw_item["stem"],
                "options": raw_item["options"],
                "high_option": raw_item["high_option"],
                "prompt_template": tmpl,
                "prefill": prefill,
            })
            if "fc_pair" in fa_blocks:
                option_by_label = {o["label"]: o["text"] for o in raw_item["options"]}
                column_defs.append({
                    "col_id": item_id,
                    "item_id": item_id,
                    "block": "fc_pair",
                    "dimension": raw_item["axis"],
                    "text": (
                        f'[{raw_item["axis"]}] {raw_item["stem"]} | '
                        f'A: {option_by_label.get("A", "")} | '
                        f'B: {option_by_label.get("B", "")}'
                    ),
                    "encoding": (
                        "+1=high,-1=low" if fc_pair_sign_alignment else "+1=A,-1=B"
                    ),
                    # Letter whose selection maps to matrix +1 for this item.
                    # With alignment on this is the axis-high pole; with
                    # alignment off it's always "A" (raw letter encoding).
                    "high_option": raw_item["high_option"] if fc_pair_sign_alignment else "A",
                    # Always retain the underlying pole assignment so downstream
                    # diagnostics / re-alignment remain possible, but under a
                    # separate key that the labeller description does not read.
                    "axis_high_letter": raw_item["high_option"],
                })
        return items, column_defs

    # ── Hybrid format (v3+) ──────────────────────────────────────────────
    items = []
    column_defs = []

    # ── Block 1: Forced choice ────────────────────────────────────────────
    for pair in data["block_1_forced_choice"]["pairs"]:
        items.append({
            "id": pair["id"],
            "type": "forced_choice",
            "block": 1,
            "option_a": pair["option_a"],
            "option_b": pair["option_b"],
        })
        if "fc" in fa_blocks:
            column_defs.append({
                "col_id": pair["id"],
                "item_id": pair["id"],
                "block": "fc",
                "text": (
                    f'A: {pair["option_a"]["text"]} | '
                    f'B: {pair["option_b"]["text"]}'
                ),
                "encoding": "+1=A,-1=B",
            })

    # ── Block 2: Vignettes ────────────────────────────────────────────────
    for vig in data["block_2_vignettes"]["scenarios"]:
        items.append({
            "id": vig["id"],
            "type": "vignette",
            "block": 2,
            "title": vig["title"],
            "scenario": vig["scenario"],
            "options": vig["options"],
            "primary_dimensions": vig["primary_dimensions"],
        })
        if "vignette" in fa_blocks:
            dims_in_vig: set[str] = set()
            for opt in vig["options"]:
                if "scoring" not in opt:
                    raise ValueError(
                        f'Vignette option {vig["id"]}/{opt.get("label", "?")} is missing '
                        '"scoring", but "vignette" is enabled in fa_blocks.'
                    )
                for dim, score in opt["scoring"].items():
                    if score != 0:
                        dims_in_vig.add(dim)
            for dim in sorted(dims_in_vig):
                column_defs.append({
                    "col_id": f'{vig["id"]}_{dim}',
                    "item_id": vig["id"],
                    "block": "vignette",
                    "dimension": dim,
                    "text": f'[{vig["title"]}] → {dim}',
                    "encoding": "option_score",
                })

    # ── Block 3: Likert ───────────────────────────────────────────────────
    for raw_item in data["block_3_likert"]["items"]:
        item = _normalize_likert_item(raw_item)
        items.append(item)
        if "likert" in fa_blocks:
            column_defs.append({
                "col_id": item["id"],
                "item_id": item["id"],
                "block": "likert",
                "dimension": item["primary_dimension"],
                "text": item["text"],
                "encoding": "1-5",
                "reverse_keyed": item.get("reverse_keyed", False),
            })

    # ── Block 4: TRAIT MCQ (optional, benchmark-backed) ──────────────────
    for raw_item in data.get("block_4_trait_mcq", {}).get("items", []):
        items.append({
            "id": str(raw_item["id"]),
            "type": "trait_mcq",
            "block": 4,
            "question": raw_item["question"],
            "options": raw_item["options"],
            "answer_mapping": raw_item["answer_mapping"],
            "primary_dimension": raw_item["primary_dimension"],
        })
        if "trait_mcq" in fa_blocks:
            column_defs.append({
                "col_id": str(raw_item["id"]),
                "item_id": str(raw_item["id"]),
                "block": "trait_mcq",
                "dimension": raw_item["primary_dimension"],
                "text": raw_item["question"],
                "encoding": "letter_1-4",
            })

    return items, column_defs
