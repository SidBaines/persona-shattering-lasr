#!/usr/bin/env python3
"""Gemini self-consistency experiment — old adapter only, temp=0.5, 3 repeats.

Cheap targeted experiment: run only the `old` adapter with only Gemini Flash 2.0
at temperature=0.5 with 3 repeats. This gives a direct measure of how consistent
Gemini is when its sampling temperature is non-zero.

Run for both neuroticism and coherence metrics.

Usage::

    # Both metrics (default)
    uv run python -m scripts_dev.rollout_experiments.neuroticism.gemini_consistency_experiment

    # Single metric
    uv run python -m scripts_dev.rollout_experiments.neuroticism.gemini_consistency_experiment --metric neuroticism
    uv run python -m scripts_dev.rollout_experiments.neuroticism.gemini_consistency_experiment --metric coherence

    # Dry run
    uv run python -m scripts_dev.rollout_experiments.neuroticism.gemini_consistency_experiment --dry-run

    # With HF upload
    uv run python -m scripts_dev.rollout_experiments.neuroticism.gemini_consistency_experiment --upload

Cost estimate (old adapter, 1700 responses, 1 rater, 3 repeats):
    ~5,100 Gemini Flash 2.0 calls × 2 metrics = ~10,200 calls total
    At ~$0.10/1M tokens → well under $1 total.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from src_dev.persona_metrics.llm_judge_agreement import (
    OceanJudgeRunConfig,
    JudgeRaterConfig,
    build_judge_run_key,
    get_judge_run_dir,
    run_ocean_judge_run,
)
from src_dev.persona_metrics.config import JudgeLLMConfig
from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait
from src_dev.utils.hf_hub import login_from_env, upload_folder_to_dataset_repo

# ── Config ─────────────────────────────────────────────────────────────────────

JUDGE_DATASETS: dict[str, Path] = {
    "neuroticism": Path("scratch/judge_datasets/neuroticism_old_sweep.jsonl"),
    "coherence":   Path("scratch/judge_datasets/neuroticism_old_sweep.jsonl"),
}

HF_REPO_ID = "persona-shattering-lasr/ocean_judge_runs"
HF_PATH_PREFIX = "gemini_consistency"

TEMPERATURE = 0.5
JUDGE_REPEATS = 3
ADAPTER = "old"

# ── Rater ──────────────────────────────────────────────────────────────────────


def _make_rater(metric_name: str) -> JudgeRaterConfig:
    return JudgeRaterConfig(
        rater_id="gemini_flash_20",
        metric_name=metric_name,
        judge=JudgeLLMConfig(
            provider="openrouter",
            model="google/gemini-2.0-flash-001",
            temperature=TEMPERATURE,
            max_concurrent=10,
        ),
    )


# ── Upload ─────────────────────────────────────────────────────────────────────


def upload_run(config: OceanJudgeRunConfig, metric: str) -> str:
    login_from_env()
    judge_dir = get_judge_run_dir(config)
    judge_key = build_judge_run_key(config)
    hf_path = f"{HF_PATH_PREFIX}/{metric}/{ADAPTER}/{judge_key}"
    return upload_folder_to_dataset_repo(
        local_dir=judge_dir,
        repo_id=HF_REPO_ID,
        path_in_repo=hf_path,
        commit_message=f"Upload gemini consistency run {judge_key} ({metric}/{ADAPTER})",
    )


# ── Main ───────────────────────────────────────────────────────────────────────


def run_metric(metric: str, upload: bool) -> dict:
    dataset_path = JUDGE_DATASETS[metric]
    if not dataset_path.exists():
        raise FileNotFoundError(f"Judge dataset not found: {dataset_path}")

    rater = _make_rater(metric)

    print(f"\n{'='*60}")
    print(f"  Gemini consistency: metric={metric}  adapter={ADAPTER}")
    print(f"  Dataset    : {dataset_path}")
    print(f"  Rater      : {rater.rater_id}  model=google/gemini-2.0-flash-001")
    print(f"  Temperature: {TEMPERATURE}  Repeats: {JUDGE_REPEATS}")
    print(f"  Est. calls : 1700 × {JUDGE_REPEATS} = {1700 * JUDGE_REPEATS:,}")
    print(f"{'='*60}\n")

    config = OceanJudgeRunConfig(
        trait=OceanTrait.neuroticism,
        dataset_path=dataset_path,
        judge_raters=[rater],
        judge_repeats=JUDGE_REPEATS,
        upload=False,
    )
    result = run_ocean_judge_run(config)

    agreement = result.get("analysis", {}).get("agreement", {})
    print(f"\n  judge_key      : {result['judge_key']}")
    print(f"  judge_dir      : {result['judge_dir']}")
    print(f"  responses      : {result['num_responses']}")
    print(f"  Krippendorff α : {agreement.get('ordinal_krippendorff_alpha', float('nan')):.3f}")
    print(f"  QWK            : {agreement.get('mean_pairwise_qwk', float('nan')):.3f}")

    if upload:
        url = upload_run(config, metric)
        result["upload_url"] = url
        print(f"  Uploaded to    : {url}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gemini consistency experiment (old adapter, temp=0.5, 3 repeats)."
    )
    parser.add_argument(
        "--metric", choices=["neuroticism", "coherence", "both"], default="both",
        help="Which metric to run (default: both).",
    )
    parser.add_argument("--upload", action="store_true",
                        help=f"Upload results to HF ({HF_REPO_ID}).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config without making API calls.")
    args = parser.parse_args()

    load_dotenv()

    metrics = ["neuroticism", "coherence"] if args.metric == "both" else [args.metric]

    if args.dry_run:
        print(f"\nDRY RUN — Gemini consistency experiment")
        print(f"  Adapter    : {ADAPTER}")
        print(f"  Metrics    : {metrics}")
        print(f"  Rater      : gemini_flash_20 (google/gemini-2.0-flash-001)")
        print(f"  Temperature: {TEMPERATURE}")
        print(f"  Repeats    : {JUDGE_REPEATS}")
        print(f"  Est. calls : {1700 * JUDGE_REPEATS * len(metrics):,}")
        print(f"  Upload     : {args.upload} -> {HF_REPO_ID}/{HF_PATH_PREFIX}/<metric>/old/<judge_key>")
        for m in metrics:
            print(f"  [{m}] {JUDGE_DATASETS[m]}")
        return

    results: dict[str, dict] = {}
    for metric in metrics:
        results[metric] = run_metric(metric, args.upload)

    print("\n\nDone.")
    for metric, result in results.items():
        url = result.get("upload_url", "not uploaded")
        print(f"  [{metric}] {result['judge_dir']}  ({url})")


if __name__ == "__main__":
    main()
