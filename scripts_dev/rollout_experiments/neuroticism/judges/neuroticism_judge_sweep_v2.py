#!/usr/bin/env python3
"""Neuroticism judge sweep v2 — temperature=0.5, 3 repeats, 3 raters.

Runs GPT-4o-mini, Gemini Flash 2.0, and Gemini Flash 1.5 8B against the
pre-generated neuroticism judge datasets with temperature=0.5 to measure
self-consistency and enable cross-model comparison.

Usage::

    # All 4 adapters
    uv run python -m scripts_dev.rollout_experiments.neuroticism.neuroticism_judge_sweep_v2

    # Single adapter
    uv run python -m scripts_dev.rollout_experiments.neuroticism.neuroticism_judge_sweep_v2 sft

    # Dry run
    uv run python -m scripts_dev.rollout_experiments.neuroticism.neuroticism_judge_sweep_v2 --dry-run

    # With HF upload
    uv run python -m scripts_dev.rollout_experiments.neuroticism.neuroticism_judge_sweep_v2 --upload
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

# ── Judge datasets ─────────────────────────────────────────────────────────────

JUDGE_DATASETS: dict[str, Path] = {
    "sft":  Path("scratch/judge_datasets/neuroticism_sft_sweep.jsonl"),
    "dpo":  Path("scratch/judge_datasets/neuroticism_dpo_sweep.jsonl"),
    "soup": Path("scratch/judge_datasets/neuroticism_soup_sweep.jsonl"),
    "old":  Path("scratch/judge_datasets/neuroticism_old_sweep.jsonl"),
}

HF_REPO_ID = "persona-shattering-lasr/ocean_judge_runs"
HF_PATH_PREFIX = "neuroticism_lora_sweep_v2"

TEMPERATURE = 0.5
JUDGE_REPEATS = 3

# ── Rater panel ────────────────────────────────────────────────────────────────

_RATERS: list[JudgeRaterConfig] = [
    JudgeRaterConfig(
        rater_id="gpt_4o_mini",
        metric_name="neuroticism",
        judge=JudgeLLMConfig(
            provider="openrouter",
            model="openai/gpt-4o-mini",
            temperature=TEMPERATURE,
            max_concurrent=10,
        ),
    ),
    JudgeRaterConfig(
        rater_id="gemini_flash_20",
        metric_name="neuroticism",
        judge=JudgeLLMConfig(
            provider="openrouter",
            model="google/gemini-2.0-flash-001",
            temperature=TEMPERATURE,
            max_concurrent=10,
        ),
    ),
    JudgeRaterConfig(
        rater_id="gemini_flash_15_8b",
        metric_name="neuroticism",
        judge=JudgeLLMConfig(
            provider="openrouter",
            model="google/gemini-flash-1.5-8b",
            temperature=TEMPERATURE,
            max_concurrent=10,
        ),
    ),
]

# ── Upload ─────────────────────────────────────────────────────────────────────


def upload_run(config: OceanJudgeRunConfig, dataset_stem: str) -> str:
    login_from_env()
    judge_dir = get_judge_run_dir(config)
    judge_key = build_judge_run_key(config)
    hf_path = f"{HF_PATH_PREFIX}/{dataset_stem}/{judge_key}"
    return upload_folder_to_dataset_repo(
        local_dir=judge_dir,
        repo_id=HF_REPO_ID,
        path_in_repo=hf_path,
        commit_message=f"Upload neuroticism v2 judge run {judge_key} ({dataset_stem})",
    )


# ── Main ───────────────────────────────────────────────────────────────────────


def run_adapter(name: str, dataset_path: Path, upload: bool) -> dict:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Judge dataset not found: {dataset_path}")

    print(f"\n{'='*60}")
    print(f"  Neuroticism judge v2: {name}  ({dataset_path})")
    print(f"  Raters     : {[r.rater_id for r in _RATERS]}")
    print(f"  Repeats    : {JUDGE_REPEATS}  Temperature: {TEMPERATURE}")
    print(f"{'='*60}\n")

    config = OceanJudgeRunConfig(
        trait=OceanTrait.neuroticism,
        dataset_path=dataset_path,
        judge_raters=_RATERS,
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
        url = upload_run(config, Path(str(dataset_path)).stem)
        result["upload_url"] = url
        print(f"  Uploaded to    : {url}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Neuroticism judge sweep v2 (temp=0.5, 3 repeats, 3 raters)."
    )
    parser.add_argument(
        "adapters", nargs="*", default=[],
        help=f"Adapter(s) to run. Defaults to all: {list(JUDGE_DATASETS.keys())}",
    )
    parser.add_argument("--upload", action="store_true",
                        help=f"Upload results to HF ({HF_REPO_ID}).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config without making API calls.")
    args = parser.parse_args()

    load_dotenv()

    requested = args.adapters or list(JUDGE_DATASETS.keys())
    unknown = [a for a in requested if a not in JUDGE_DATASETS]
    if unknown:
        print(f"Unknown adapter(s): {unknown}. Valid: {list(JUDGE_DATASETS.keys())}")
        sys.exit(1)

    if args.dry_run:
        print(f"\nDRY RUN — Neuroticism judge sweep v2")
        print(f"  Adapters   : {requested}")
        print(f"  Raters     : {[r.rater_id for r in _RATERS]}")
        print(f"  Temperature: {TEMPERATURE}")
        print(f"  Repeats    : {JUDGE_REPEATS}")
        print(f"  Est. calls : {len(requested) * 1700 * len(_RATERS) * JUDGE_REPEATS:,}")
        print(f"  Upload     : {args.upload} -> {HF_REPO_ID}/{HF_PATH_PREFIX}/<dataset>/<judge_key>")
        for name in requested:
            print(f"  [{name}] {JUDGE_DATASETS[name]}")
        return

    results: dict[str, dict] = {}
    for name in requested:
        results[name] = run_adapter(name, JUDGE_DATASETS[name], args.upload)

    print("\n\nAll adapters done.")
    for name, result in results.items():
        url = result.get("upload_url", "not uploaded")
        print(f"  [{name}] {result['judge_dir']}  ({url})")


if __name__ == "__main__":
    main()
