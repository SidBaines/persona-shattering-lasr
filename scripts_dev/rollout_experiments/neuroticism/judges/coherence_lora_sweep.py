#!/usr/bin/env python3
"""Coherence judge sweep for the 4 neuroticism LoRA adapters.

Runs the coherence judge panel against pre-generated judge datasets
(one per adapter), producing per-scale coherence scores that can be
plotted alongside the neuroticism trait scores.

Usage::

    uv run python -m scripts_dev.rollout_experiments.neuroticism.coherence_lora_sweep

To run a single adapter::

    uv run python -m scripts_dev.rollout_experiments.neuroticism.coherence_lora_sweep sft
    uv run python -m scripts_dev.rollout_experiments.neuroticism.coherence_lora_sweep dpo soup

To run only Gemini (skip GPT)::

    uv run python -m scripts_dev.rollout_experiments.neuroticism.coherence_lora_sweep --gemini-only

Dry run (print config, no API calls)::

    uv run python -m scripts_dev.rollout_experiments.neuroticism.coherence_lora_sweep --dry-run

Upload results to HF after each adapter::

    uv run python -m scripts_dev.rollout_experiments.neuroticism.coherence_lora_sweep --upload
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
HF_PATH_PREFIX = "coherence_lora_sweep"

# ── Rater panel ────────────────────────────────────────────────────────────────

_ALL_RATERS: list[JudgeRaterConfig] = [
    JudgeRaterConfig(
        rater_id="gpt_4o_mini",
        metric_name="coherence",
        judge=JudgeLLMConfig(
            provider="openrouter",
            model="openai/gpt-4o-mini",
            temperature=0.0,
            max_concurrent=10,
        ),
    ),
    JudgeRaterConfig(
        rater_id="gemini_flash_20",
        metric_name="coherence",
        judge=JudgeLLMConfig(
            provider="openrouter",
            model="google/gemini-2.0-flash-001",
            temperature=0.0,
            max_concurrent=10,
        ),
    ),
]

JUDGE_REPEATS = 3

# ── Upload ─────────────────────────────────────────────────────────────────────


def upload_coherence_run(config: OceanJudgeRunConfig) -> str:
    """Upload one coherence judge run dir to HF.

    Uses a flat path under ``coherence_lora_sweep/`` since the datasets
    live outside the standard ocean pipeline run-dir structure.
    """
    login_from_env()
    judge_dir = get_judge_run_dir(config)
    judge_key = build_judge_run_key(config)
    # dataset stem, e.g. neuroticism_sft_sweep
    dataset_stem = Path(str(config.dataset_path)).stem
    hf_path = f"{HF_PATH_PREFIX}/{dataset_stem}/{judge_key}"
    return upload_folder_to_dataset_repo(
        local_dir=judge_dir,
        repo_id=HF_REPO_ID,
        path_in_repo=hf_path,
        commit_message=f"Upload coherence sweep judge run {judge_key} ({dataset_stem})",
    )


# ── Main ───────────────────────────────────────────────────────────────────────


def run_adapter(
    name: str,
    dataset_path: Path,
    raters: list[JudgeRaterConfig],
    judge_repeats: int,
    upload: bool,
) -> dict:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Judge dataset not found: {dataset_path}")

    print(f"\n{'='*60}")
    print(f"  Coherence sweep: {name}  ({dataset_path})")
    print(f"  Raters: {[r.rater_id for r in raters]}")
    print(f"  Repeats: {judge_repeats}")
    print(f"{'='*60}\n")

    config = OceanJudgeRunConfig(
        trait=OceanTrait.neuroticism,  # placeholder — unused during judge scoring
        dataset_path=dataset_path,
        judge_raters=raters,
        judge_repeats=judge_repeats,
        upload=False,  # we handle upload ourselves with the right HF path
    )
    result = run_ocean_judge_run(config)

    agreement = result.get("analysis", {}).get("agreement", {})
    print(f"\n  judge_key         : {result['judge_key']}")
    print(f"  judge_dir         : {result['judge_dir']}")
    print(f"  responses         : {result['num_responses']}")
    print(f"  Krippendorff α    : {agreement.get('ordinal_krippendorff_alpha', float('nan')):.3f}")
    print(f"  QWK               : {agreement.get('mean_pairwise_qwk', float('nan')):.3f}")

    if upload:
        url = upload_coherence_run(config)
        result["upload_url"] = url
        print(f"  Uploaded to       : {url}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Coherence judge sweep for neuroticism LoRA adapters."
    )
    parser.add_argument(
        "adapters",
        nargs="*",
        default=[],
        help=f"Adapter(s) to run. Defaults to all: {list(JUDGE_DATASETS.keys())}",
    )
    parser.add_argument(
        "--gemini-only",
        action="store_true",
        help="Run only the gemini_flash_20 rater (skip gpt_4o_mini).",
    )
    parser.add_argument(
        "--judge-repeats",
        type=int,
        default=JUDGE_REPEATS,
        help=f"Judge scoring repeats per response per rater (default: {JUDGE_REPEATS}).",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help=f"Upload results to HF ({HF_REPO_ID}) after each adapter.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config summary without making any API calls.",
    )
    args = parser.parse_args()

    load_dotenv()

    requested = args.adapters or list(JUDGE_DATASETS.keys())
    unknown = [a for a in requested if a not in JUDGE_DATASETS]
    if unknown:
        print(f"Unknown adapter(s): {unknown}. Valid: {list(JUDGE_DATASETS.keys())}")
        sys.exit(1)

    raters = [r for r in _ALL_RATERS if not (args.gemini_only and r.rater_id == "gpt_4o_mini")]

    if args.dry_run:
        print(f"\nDRY RUN — Coherence sweep")
        print(f"  Adapters   : {requested}")
        print(f"  Raters     : {[r.rater_id for r in raters]}")
        print(f"  Repeats    : {args.judge_repeats}")
        print(f"  Upload     : {args.upload} -> {HF_REPO_ID}/{HF_PATH_PREFIX}/<dataset>/<judge_key>")
        for name in requested:
            print(f"  [{name}] dataset: {JUDGE_DATASETS[name]}")
        return

    results: dict[str, dict] = {}
    for name in requested:
        results[name] = run_adapter(name, JUDGE_DATASETS[name], raters, args.judge_repeats, args.upload)

    print("\n\nAll adapters done.")
    for name, result in results.items():
        url = result.get("upload_url", "not uploaded")
        print(f"  [{name}] {result['judge_dir']}  ({url})")


if __name__ == "__main__":
    main()
