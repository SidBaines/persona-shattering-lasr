#!/usr/bin/env python3
"""OCEAN trait judge agreement harness — calibration dataset generation and multi-model rating.

Generates a calibration dataset for one or all OCEAN traits by running an assistant model under
three system-prompt conditions (neutral, high-trait, low-trait) on Assistant Axis prompts, then
scoring responses with multiple LLM judge raters.

Pipeline:
  1. Sample N prompts from the Assistant Axis dataset.
  2. Generate R responses per prompt under each of 3 conditions.
  3. Score each response with M judge models × K repeats.
  4. Compute inter-rater agreement (Krippendorff α, pairwise QWK, MAE).
  5. Write outputs to scratch/ocean_judge_runs/<run_key>/.

Usage:
    uv run python scripts/experiments/ocean_judge_calibration.py --trait neuroticism
    uv run python scripts/experiments/ocean_judge_calibration.py --trait all
    uv run python scripts/experiments/ocean_judge_calibration.py --trait agreeableness --dry-run
    uv run python scripts/experiments/ocean_judge_calibration.py \\
        --trait neuroticism --max-prompts 5 --responses-per-prompt 1

Outputs:
    scratch/ocean_judge_runs/<run_key>/
        prompts/source_prompts.jsonl
        responses/{neutral,high_<trait>,low_<trait>}/
        exports/all_responses.jsonl   <- inspect with jsonl_tui
        judge_calls/raw/<rater_id>.jsonl
        analysis/summary.json         <- Krippendorff α, pairwise QWK
        analysis/condition_metrics.json
        plots/

Inspect responses:
    uv run python src_dev/jsonl_tui/cli.py \\
        scratch/ocean_judge_runs/<run_key>/exports/all_responses.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from src_dev.inference import InferenceConfig
from src_dev.inference.config import GenerationConfig
from src_dev.persona_metrics.config import JudgeLLMConfig
from src_dev.persona_metrics.llm_judge_agreement import (
    JudgeRaterConfig,
    OceanJudgeAgreementConfig,
    build_ocean_system_prompts,
    build_run_key_ocean,
    get_run_dir_ocean,
    run_ocean_judge_agreement,
)
from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait

# ---------------------------------------------------------------------------
# Default judge rater panel — add more models here with one line each
# ---------------------------------------------------------------------------

_DEFAULT_RATERS = [
    JudgeRaterConfig(
        rater_id="gpt_4o_mini",
        judge=JudgeLLMConfig(
            provider="openrouter",
            model="openai/gpt-4o-mini",
            temperature=0.0,
            max_concurrent=10,
        ),
    ),
    JudgeRaterConfig(
        rater_id="haiku_35",
        judge=JudgeLLMConfig(
            provider="openrouter",
            model="anthropic/claude-3.5-haiku",
            temperature=0.0,
            max_concurrent=10,
        ),
    ),
    JudgeRaterConfig(
        rater_id="gemini_flash_20",
        judge=JudgeLLMConfig(
            provider="openrouter",
            model="google/gemini-2.0-flash-001",
            temperature=0.0,
            max_concurrent=10,
        ),
    ),
]

# ---------------------------------------------------------------------------
# Default assistant model for generating calibration responses
# ---------------------------------------------------------------------------

_DEFAULT_ASSISTANT = InferenceConfig(
    provider="openrouter",
    model="openai/gpt-4o-mini",
    generation=GenerationConfig(temperature=0.9),
)


def make_config(
    trait: OceanTrait,
    *,
    max_prompts: int = 20,
    responses_per_prompt: int = 3,
    judge_repeats: int = 3,
    raters: list[JudgeRaterConfig] | None = None,
    upload: bool = False,
) -> OceanJudgeAgreementConfig:
    """Build an OceanJudgeAgreementConfig for a single trait.

    Args:
        trait: Which OCEAN trait to run.
        max_prompts: Number of prompts to sample from the dataset.
        responses_per_prompt: Responses to generate per prompt per condition.
        judge_repeats: How many times each rater scores each response.
        raters: Judge rater panel. Defaults to ``_DEFAULT_RATERS``.
        upload: Whether to upload results to HF after the run.

    Returns:
        Fully configured ``OceanJudgeAgreementConfig``.
    """
    # Set the correct metric_name for each rater's trait-specific v2 judge
    resolved_raters = []
    for rater in (raters or _DEFAULT_RATERS):
        resolved_raters.append(
            rater.model_copy(update={"metric_name": trait.v2_metric_name})
        )
    return OceanJudgeAgreementConfig(
        trait=trait,
        assistant_inference=_DEFAULT_ASSISTANT,
        judge_raters=resolved_raters,
        max_prompts=max_prompts,
        responses_per_prompt=responses_per_prompt,
        judge_repeats=judge_repeats,
        upload=upload,
    )


def _print_dry_run(config: OceanJudgeAgreementConfig) -> None:
    """Print a dry-run summary without making any API calls."""
    run_key = build_run_key_ocean(config)
    run_dir = get_run_dir_ocean(config)
    conditions = build_ocean_system_prompts(config.trait)

    print(f"\n{'=' * 70}")
    print(f"DRY RUN — {config.trait.value.upper()} judge calibration")
    print(f"{'=' * 70}")
    print(f"  Run key      : {run_key}")
    print(f"  Output dir   : {run_dir}")
    print(f"  Prompts      : {config.max_prompts}")
    print(f"  Responses/p  : {config.responses_per_prompt}")
    print(f"  Judge repeats: {config.judge_repeats}")
    print(f"  Raters       : {[r.rater_id for r in config.judge_raters]}")
    print(f"  Metric names : {[r.metric_name for r in config.judge_raters]}")
    total_responses = config.max_prompts * config.responses_per_prompt * len(conditions)
    total_calls = total_responses * config.judge_repeats * len(config.judge_raters)
    print(f"\n  Conditions   : {list(conditions)}")
    print(f"  Total responses to generate : ~{total_responses}")
    print(f"  Total judge calls           : ~{total_calls}")
    print(f"\n  System prompts (first 120 chars each):")
    for name, prompt in conditions.items():
        print(f"    [{name}] {prompt[:120].replace(chr(10), ' ')}...")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OCEAN trait judge agreement calibration harness."
    )
    parser.add_argument(
        "--trait",
        choices=[t.value for t in OceanTrait] + ["all"],
        required=True,
        help="Which trait to run, or 'all' for all five.",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=20,
        help="Number of prompts to sample (default: 20).",
    )
    parser.add_argument(
        "--responses-per-prompt",
        type=int,
        default=3,
        help="Responses to generate per prompt per condition (default: 3).",
    )
    parser.add_argument(
        "--judge-repeats",
        type=int,
        default=3,
        help="Judge scoring repeats per response per rater (default: 3).",
    )
    parser.add_argument(
        "--mode",
        choices=["run", "analyze_only", "upload_only"],
        default="run",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload results to HF after the run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config summary without making any API calls.",
    )
    args = parser.parse_args()

    load_dotenv()

    traits = list(OceanTrait) if args.trait == "all" else [OceanTrait(args.trait)]

    for trait in traits:
        config = make_config(
            trait,
            max_prompts=args.max_prompts,
            responses_per_prompt=args.responses_per_prompt,
            judge_repeats=args.judge_repeats,
            upload=args.upload,
        )

        if args.dry_run:
            _print_dry_run(config)
            continue

        print(f"\nRunning {trait.value} judge calibration ...")
        result = run_ocean_judge_agreement(config, mode=args.mode)

        print(f"\n  run_key  : {result['run_key']}")
        print(f"  run_dir  : {result['run_dir']}")
        if "num_prompts" in result:
            print(f"  prompts  : {result['num_prompts']}")
        if "num_responses" in result:
            print(f"  responses: {result['num_responses']}")
        if "analysis" in result:
            agreement = result["analysis"].get("agreement", {})
            print(f"  Krippendorff α : {agreement.get('ordinal_krippendorff_alpha', 'n/a'):.3f}")
            print(f"  Mean pairwise QWK : {agreement.get('mean_pairwise_qwk', 'n/a'):.3f}")
        if "upload_url" in result:
            print(f"  Uploaded → {result['upload_url']}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
