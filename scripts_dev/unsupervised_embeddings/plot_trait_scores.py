"""Compute and plot per-persona TRAIT scores for a stage-2 questionnaire run.

Reads the stage-2 questionnaire output (``raw_responses.jsonl`` + the
questionnaire JSON used to generate it) and writes:

- ``trait_scores.csv``, ``trait_scores_coverage.csv``, ``trait_scores_summary.json``
- ``trait_histograms.pdf``, ``trait_heatmap.pdf``
- ``trait_vs_factor_correlations.pdf`` (only when an FA result is found)

Usage:
    uv run python scripts_dev/unsupervised_embeddings/plot_trait_scores.py \\
        --questionnaire-dir scratch/psychometric_fa/<run_id>/questionnaire \\
        --questionnaire-json datasets/psychometric_questionnaires/trait_ocean_v1.json \\
        [--fa-result scratch/psychometric_fa/<run_id>/factor_analysis/<subdir>]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src_dev.factor_analysis.trait_score_plots import plot_all_trait_plots
from src_dev.factor_analysis.trait_scoring import compute_trait_scores, save_trait_scores


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questionnaire-dir",
        type=Path,
        required=True,
        help="Directory containing raw_responses.jsonl (stage-2 output).",
    )
    parser.add_argument(
        "--questionnaire-json",
        type=Path,
        required=True,
        help="Path to the questionnaire JSON used for the run.",
    )
    parser.add_argument(
        "--fa-result",
        type=Path,
        default=None,
        help="Optional base path to a factor-analysis result (no extension).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write scores + plots. Default: <questionnaire-dir>/trait_scores",
    )
    parser.add_argument(
        "--title-prefix",
        type=str,
        default=None,
        help="Optional title prefix for plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    q_dir: Path = args.questionnaire_dir
    raw_path = q_dir / "raw_responses.jsonl"
    if not raw_path.exists():
        raise FileNotFoundError(f"raw_responses.jsonl not found in {q_dir}")

    output_dir = args.output_dir or (q_dir / "trait_scores")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load stage-2 metadata if available so CSV can carry sample_id / group.
    metadata: list[dict] | None = None
    meta_path = q_dir / "metadata.jsonl"
    if meta_path.exists():
        metadata = [json.loads(line) for line in meta_path.open() if line.strip()]

    result = compute_trait_scores(
        raw_responses_path=raw_path,
        questionnaire_path=args.questionnaire_json,
    )
    written = save_trait_scores(result, output_dir, metadata=metadata)
    print(f"[trait-scores] Wrote {len(written)} CSV/JSON files to {output_dir}")

    factor_scores = None
    factor_labels = None
    if args.fa_result is not None:
        from src_dev.factor_analysis.persistence import load_factor_analysis

        fa = load_factor_analysis(args.fa_result)
        factor_scores = fa["scores"]
        factor_labels = [f"F{i+1}" for i in range(factor_scores.shape[1])]

    plot_paths = plot_all_trait_plots(
        result.scores,
        output_dir,
        factor_scores=factor_scores,
        factor_labels=factor_labels,
        title_prefix=args.title_prefix or q_dir.parent.name,
    )
    for name, path in plot_paths.items():
        print(f"[trait-scores] {name}: {path}")


if __name__ == "__main__":
    main()
