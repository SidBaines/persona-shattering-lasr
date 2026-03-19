"""Calibrate neuroticism_v2 against human ratings on the heldout calibration split."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from src_dev.persona_metrics.config import JudgeLLMConfig
from src_dev.persona_metrics.judge_calibration import (
    aggregate_reference_from_csvs,
    author_scores,
    category_breakdown,
    filter_items,
    load_calibration_items,
    load_reference_set,
    reference_scores,
    summarize_inter_rater,
    summarize_pair,
)
from src_dev.persona_metrics.metrics.ocean_v2 import NeuroticismV2Evaluation

DEFAULT_DATASET = Path("scratch/judge_calibration/neuroticism_v2/neuroticism_v2_calibration.jsonl")


async def score_model(
    items,
    config: JudgeLLMConfig,
) -> list[int | None]:
    judge = NeuroticismV2Evaluation(judge_config=config)
    responses = [item.response for item in items]
    questions = [item.question for item in items]
    results = await judge.evaluate_batch_async(responses, questions)
    score_key = f"{judge.name}.score"
    return [None if result.get(score_key) == judge.score_error else result.get(score_key) for result in results]


def _print_summary(title: str, stats: dict[str, float | int]) -> None:
    print(title)
    print(f"  n={stats['n']} pearson={stats['pearson']:.3f} spearman={stats['spearman']:.3f} mae={stats['mae']:.3f} within_one={stats['within_one']:.3f} exact={stats['exact']:.3f}")


def maybe_plot(output_dir: Path, model_name: str, human_reference, scores) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    valid = [(ref, score) for ref, score in zip(human_reference, scores) if ref is not None and score is not None]
    if not valid:
        return
    xs = [left for left, _ in valid]
    ys = [right for _, right in valid]
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(xs, ys, alpha=0.8)
    ax.plot([-4, 4], [-4, 4], linestyle="--", linewidth=1)
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)
    ax.set_xlabel("Human median score")
    ax.set_ylabel(f"{model_name} score")
    ax.set_title(f"neuroticism_v2: {model_name}")
    ax.grid(alpha=0.2)
    fig.savefig(output_dir / f"{model_name.replace('/', '_')}_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate neuroticism_v2 against human ratings.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", choices=["dev", "heldout"], default="heldout")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--provider", choices=["openai", "openrouter", "anthropic"], default="openrouter")
    parser.add_argument("--reference-json", type=Path, default=None)
    parser.add_argument("--ratings-csv", nargs="+", type=Path, default=None)
    parser.add_argument("--max-concurrent", type=int, default=20)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("scratch/judge_calibration/neuroticism_v2"))
    args = parser.parse_args()

    if args.reference_json is None and not args.ratings_csv:
        raise ValueError("Provide --reference-json or --ratings-csv for the human reference.")

    load_dotenv()

    items = filter_items(load_calibration_items(args.dataset), split=args.split)
    if args.reference_json is not None:
        reference_set = load_reference_set(args.reference_json)
    else:
        reference_set = aggregate_reference_from_csvs(args.ratings_csv, items, require_complete=False)

    human_reference = reference_scores(reference_set)
    author_reference = author_scores(items)
    inter_rater = summarize_inter_rater(reference_set)

    report: dict[str, object] = {
        "trait": "neuroticism_v2",
        "split": args.split,
        "num_items": len(items),
        "raters": reference_set.raters,
        "inter_rater": inter_rater,
        "models": {},
    }

    print(f"Items: {len(items)}  split={args.split}")
    print(f"Raters: {reference_set.raters}")
    if inter_rater.get("summary"):
        print("Inter-rater summary:")
        print(json.dumps(inter_rater["summary"], indent=2, sort_keys=True))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        cfg = JudgeLLMConfig(
            provider=args.provider,
            model=model,
            temperature=0.0,
            max_concurrent=args.max_concurrent,
        )
        scores = asyncio.run(score_model(items, cfg))
        human_stats = summarize_pair(human_reference, scores)
        author_stats = summarize_pair(author_reference, scores)
        category_stats = category_breakdown(items, human_reference, scores)
        error_count = sum(score is None for score in scores)

        print(f"\nModel: {model}")
        _print_summary("  vs human", human_stats)
        _print_summary("  vs author", author_stats)
        print(f"  errors={error_count}/{len(scores)}")

        report["models"][model] = {
            "scores": scores,
            "errors": error_count,
            "human_alignment": human_stats,
            "author_alignment": author_stats,
            "category_breakdown": category_stats,
        }
        if args.plot:
            maybe_plot(args.output_dir, model, human_reference, scores)

    report_path = args.output_dir / "summary.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()
