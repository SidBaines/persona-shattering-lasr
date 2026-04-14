"""Compute per-persona TRAIT scores from raw questionnaire responses.

Reads the raw_responses.jsonl written by the stage-2 questionnaire loop and,
for each (persona k, TRAIT item) cell, derives a "high-trait" score using
the item's ``answer_mapping`` (``{letter: 0 or 1}`` for shuffled A/B/C/D).

When the raw entry includes a ``probs`` field (logprob mode), the score is
the continuous expectation ``sum_letter P(letter) * answer_mapping[letter]``.
Otherwise, it falls back to the binary 0/1 value derived from the parsed
categorical choice. Per-persona trait scores are the mean across that
trait's items.

This mirrors the TRAIT benchmark's canonical scoring and prefers the raw
model distribution when available, for more stable (less noisy) estimates.

Only items with ``type == "trait_mcq"`` are considered — other item types
(Likert, forced-choice, vignette) are ignored so this function can be used on
hybrid questionnaires that include TRAIT MCQs alongside other blocks.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class TraitScoringResult:
    """Per-persona TRAIT scores plus coverage diagnostics.

    Attributes:
        scores: DataFrame indexed by ``k`` (persona row index) with one column
            per trait holding mean "high-trait option" probability in [0, 1].
            NaN for traits with zero valid responses for that persona.
        coverage: DataFrame indexed by ``k`` with one column per trait
            holding the count of valid (parsed, non-null) responses for that
            persona/trait (post choice-mass filter).
        items_per_trait: Total number of questionnaire items per trait.
        trait_order: Canonical trait order used in the scores columns.
        min_trait_coverage: Minimum fraction of a trait's items that must have
            valid responses for a persona's cell to be kept. 0.0 means no
            persona filter was applied.
        filtered_by_trait: Per-trait count of personas whose cell was dropped
            to NaN because their coverage fraction fell below
            ``min_trait_coverage``. Empty when no filter was applied.
    """

    scores: pd.DataFrame
    coverage: pd.DataFrame
    items_per_trait: dict[str, int]
    trait_order: list[str]
    min_trait_coverage: float = 0.0
    filtered_by_trait: dict[str, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.filtered_by_trait is None:
            self.filtered_by_trait = {}


def _load_questionnaire_trait_mcq_items(questionnaire_path: Path) -> dict[str, dict]:
    """Load trait_mcq items from a questionnaire JSON, keyed by item id.

    Accepts either the minimal TRAIT-only format (only ``block_4_trait_mcq``)
    or the hybrid format that embeds ``block_4_trait_mcq`` alongside other
    blocks.
    """
    with open(questionnaire_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items: dict[str, dict] = {}
    for raw in data.get("block_4_trait_mcq", {}).get("items", []):
        items[str(raw["id"])] = {
            "id": str(raw["id"]),
            "answer_mapping": dict(raw["answer_mapping"]),
            "primary_dimension": raw["primary_dimension"],
        }
    return items


def compute_trait_scores(
    raw_responses_path: Path | str,
    questionnaire_path: Path | str,
    *,
    trait_order: list[str] | None = None,
    dynamic_mass_filter: bool = True,
    min_choice_mass: float = 0.0,
    min_trait_coverage: float = 0.0,
) -> TraitScoringResult:
    """Compute per-persona TRAIT scores from raw questionnaire responses.

    Args:
        raw_responses_path: Path to the stage-2 ``raw_responses.jsonl``.
        questionnaire_path: Path to the questionnaire JSON (for
            ``answer_mapping`` + ``primary_dimension`` lookup).
        trait_order: Optional explicit column order for the scores DataFrame.
            Defaults to the OCEAN traits (lower-cased) in canonical order,
            restricted to traits actually present in the questionnaire.
        dynamic_mass_filter: If True (default), drop logprob-mode responses
            whose ``choice_mass`` is below ``1 / num_choices`` (i.e. below the
            uniform-prior floor). Mirrors the default behaviour of the
            personality logprob scorer's ``logprob_mcq_ratio`` metric. Only
            applies to entries that carry a ``choice_mass`` field (logprob
            mode); categorical fallbacks are unaffected.
        min_choice_mass: Fixed minimum ``choice_mass`` threshold applied after
            the dynamic filter. Default 0.0 (no fixed filter).
        min_trait_coverage: Minimum fraction of a trait's items (in [0, 1])
            that must have valid post-filter responses for a persona's score
            to be retained. Cells below this threshold are set to NaN (so they
            drop out of downstream plots/aggregations). Default 0.0 keeps
            every cell with at least one valid response.

    Returns:
        A ``TraitScoringResult`` with per-persona score and coverage frames.
    """
    raw_responses_path = Path(raw_responses_path)
    questionnaire_path = Path(questionnaire_path)

    items = _load_questionnaire_trait_mcq_items(questionnaire_path)
    if not items:
        raise ValueError(
            f"No trait_mcq items found in {questionnaire_path}. "
            "Did you point it at a TRAIT questionnaire?"
        )

    items_per_trait: dict[str, int] = defaultdict(int)
    for it in items.values():
        items_per_trait[it["primary_dimension"]] += 1

    if trait_order is None:
        ocean = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
        trait_order = [t for t in ocean if t in items_per_trait]
        # Append any unexpected traits (e.g. dark triad) in sorted order.
        extras = sorted(set(items_per_trait) - set(trait_order))
        trait_order = trait_order + extras

    # (k, trait) -> list of per-item high-trait scores in [0, 1].
    # Continuous when logprob mode provides ``probs``; else binary 0/1.
    per_cell: dict[tuple[int, str], list[float]] = defaultdict(list)

    with open(raw_responses_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("item_type") != "trait_mcq":
                continue
            item_id = str(entry["item_id"])
            item = items.get(item_id)
            if item is None:
                continue
            mapping = item["answer_mapping"]

            probs = entry.get("probs")
            if isinstance(probs, dict) and probs:
                # Apply choice-mass filters (logprob mode only). ``choice_mass``
                # is the total probability on valid choice letters out of the
                # full vocabulary, written alongside ``probs`` by the stage-2
                # logprob parser.
                cm = entry.get("choice_mass")
                if dynamic_mass_filter and isinstance(cm, (int, float)):
                    num_choices = len(mapping) or 4
                    if cm < (1.0 / num_choices if num_choices > 0 else 0.0):
                        continue
                if (
                    min_choice_mass > 0.0
                    and isinstance(cm, (int, float))
                    and cm < min_choice_mass
                ):
                    continue
                # Continuous expected score over found letters.
                value = 0.0
                for letter, p in probs.items():
                    if str(letter) in mapping:
                        value += float(p) * float(mapping[str(letter)])
                per_cell[(int(entry["k"]), item["primary_dimension"])].append(value)
                continue

            choice = entry.get("parsed_choice")
            if choice is None:
                continue
            if str(choice) not in mapping:
                continue
            per_cell[(int(entry["k"]), item["primary_dimension"])].append(
                float(mapping[str(choice)])
            )

    if not per_cell:
        raise ValueError(
            f"No parseable trait_mcq responses found in {raw_responses_path}."
        )

    k_values = sorted({k for k, _ in per_cell})
    scores = pd.DataFrame(
        np.nan, index=pd.Index(k_values, name="k"), columns=trait_order, dtype=float
    )
    coverage = pd.DataFrame(
        0, index=pd.Index(k_values, name="k"), columns=trait_order, dtype=int
    )

    for (k, trait), values in per_cell.items():
        if trait not in scores.columns:
            continue
        scores.loc[k, trait] = float(np.mean(values))
        coverage.loc[k, trait] = int(len(values))

    filtered_by_trait: dict[str, int] = {t: 0 for t in trait_order}
    if min_trait_coverage > 0.0:
        for trait in trait_order:
            n_items = items_per_trait.get(trait, 0)
            if n_items <= 0:
                continue
            threshold = min_trait_coverage * n_items
            # Only consider cells that actually have a score (i.e. at least
            # one valid response); cells that were already NaN don't count as
            # "filtered" here.
            has_score = scores[trait].notna()
            low = has_score & (coverage[trait].astype(float) < threshold)
            filtered_by_trait[trait] = int(low.sum())
            scores.loc[low, trait] = np.nan

    return TraitScoringResult(
        scores=scores,
        coverage=coverage,
        items_per_trait=dict(items_per_trait),
        trait_order=trait_order,
        min_trait_coverage=float(min_trait_coverage),
        filtered_by_trait=filtered_by_trait,
    )


def save_trait_scores(
    result: TraitScoringResult,
    output_dir: Path | str,
    *,
    metadata: list[dict] | None = None,
) -> dict[str, Path]:
    """Save a TraitScoringResult to disk as CSVs (scores + coverage + summary).

    If ``metadata`` is provided (the stage-2 metadata list mapping k to
    sample_id etc.), a ``trait_scores_with_metadata.csv`` is also written
    joining sample_id / input_group_id onto each row.

    Returns a dict of {name: path} for the written files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}

    scores_path = output_dir / "trait_scores.csv"
    result.scores.to_csv(scores_path)
    written["scores"] = scores_path

    coverage_path = output_dir / "trait_scores_coverage.csv"
    result.coverage.to_csv(coverage_path)
    written["coverage"] = coverage_path

    summary = {
        "trait_order": result.trait_order,
        "items_per_trait": result.items_per_trait,
        "n_personas": int(len(result.scores)),
        "mean_by_trait": result.scores.mean(skipna=True).to_dict(),
        "std_by_trait": result.scores.std(skipna=True).to_dict(),
        "n_personas_scored_by_trait": result.scores.notna().sum().astype(int).to_dict(),
        "min_trait_coverage": result.min_trait_coverage,
        "n_personas_filtered_by_trait": dict(result.filtered_by_trait),
    }
    summary_path = output_dir / "trait_scores_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    written["summary"] = summary_path

    if metadata is not None:
        meta_by_k = {
            i: {"sample_id": m.get("sample_id"), "input_group_id": m.get("input_group_id")}
            for i, m in enumerate(metadata)
        }
        df = result.scores.copy()
        df.insert(0, "sample_id", df.index.map(lambda k: meta_by_k.get(int(k), {}).get("sample_id")))
        df.insert(1, "input_group_id", df.index.map(
            lambda k: meta_by_k.get(int(k), {}).get("input_group_id"),
        ))
        joined_path = output_dir / "trait_scores_with_metadata.csv"
        df.to_csv(joined_path)
        written["with_metadata"] = joined_path

    return written
