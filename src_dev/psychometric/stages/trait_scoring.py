"""Stage 2b — per-persona OCEAN trait scoring from trait_mcq items.

Thin wrapper around ``src_dev.factor_analysis.trait_scoring.compute_trait_scores``
+ ``save_trait_scores`` and ``src_dev.factor_analysis.trait_score_plots.
plot_all_trait_plots``. No-ops (and returns an empty result) when the active
questionnaire contains no ``block_4_trait_mcq`` items.

Reads stage-2 artefacts from ``cfg.ctx.effective_questionnaire_dir /
"questionnaire"``. Writes the trait scores + plots under
``.../questionnaire/trait_scores/``.
"""

from __future__ import annotations

import json
from pathlib import Path

from src_dev.psychometric.config import (
    TraitScoringStageConfig,
    TraitScoringStageResult,
)


def run_stage_trait_scoring(
    cfg: TraitScoringStageConfig,
    *,
    questionnaire_path: Path,
    dynamic_mass_filter: bool = True,
    min_choice_mass: float = 0.0,
    title_prefix: str = "",
    metadata: list[dict] | None = None,
) -> TraitScoringStageResult:
    """Compute per-persona OCEAN trait scores from trait_mcq responses.

    Args:
        cfg: Stage config (ctx + coverage thresholds).
        questionnaire_path: Path to the active questionnaire JSON.
            Used to decide whether trait_mcq is present before spinning up
            the scorer.
        dynamic_mass_filter: Drop items whose total choice-letter mass is
            below 1/num_choices (personality logprob scorer default).
        min_choice_mass: Absolute floor applied on top of the dynamic
            filter.
        title_prefix: Prefix for the generated PDF plot titles (usually the
            questionnaire run id).
        metadata: Optional stage-2 metadata list (enriches the CSV with
            ``sample_id`` / ``input_group_id``). When None, loaded from the
            questionnaire dir.
    """
    from src_dev.factor_analysis.trait_score_plots import plot_all_trait_plots
    from src_dev.factor_analysis.trait_scoring import (
        compute_trait_scores,
        save_trait_scores,
    )

    q_dir = cfg.ctx.effective_questionnaire_dir / "questionnaire"
    raw_path = q_dir / "raw_responses.jsonl"
    if not raw_path.exists():
        print("[Trait-scoring] raw_responses.jsonl not found — skipping.")
        return TraitScoringStageResult()

    with open(questionnaire_path, "r", encoding="utf-8") as f:
        qn_raw = json.load(f)
    n_trait_items = len(qn_raw.get("block_4_trait_mcq", {}).get("items", []))
    if n_trait_items == 0:
        print("[Trait-scoring] No trait_mcq items in questionnaire — skipping.")
        return TraitScoringStageResult()

    output_dir = q_dir / "trait_scores"
    output_dir.mkdir(parents=True, exist_ok=True)

    if metadata is None:
        meta_path = q_dir / "metadata.jsonl"
        if meta_path.exists():
            metadata = [json.loads(line) for line in meta_path.open() if line.strip()]

    result = compute_trait_scores(
        raw_responses_path=raw_path,
        questionnaire_path=questionnaire_path,
        dynamic_mass_filter=dynamic_mass_filter,
        min_choice_mass=min_choice_mass,
        min_trait_coverage=cfg.min_trait_coverage,
    )
    written = save_trait_scores(result, output_dir, metadata=metadata)

    plot_paths = plot_all_trait_plots(
        result.scores,
        output_dir,
        title_prefix=title_prefix,
    )
    written.update(plot_paths)

    summary = {
        "n_personas": int(len(result.scores)),
        "trait_order": result.trait_order,
        "items_per_trait": result.items_per_trait,
        "mean_by_trait": result.scores.mean(skipna=True).to_dict(),
        "n_personas_scored_by_trait": result.scores.notna().sum().astype(int).to_dict(),
    }
    print(f"[Trait-scoring] {summary}")
    if result.min_trait_coverage > 0.0:
        print(
            f"[Trait-scoring] Persona filter: dropped cells with coverage < "
            f"{result.min_trait_coverage:.0%} of items per trait — removed "
            f"{result.filtered_by_trait}"
        )
    print(f"[Trait-scoring] Wrote {len(written)} files to {output_dir}")

    trait_scores_path = output_dir / "trait_scores.csv"
    return TraitScoringStageResult(
        trait_scores_path=trait_scores_path if trait_scores_path.exists() else None,
        summary=summary,
    )
