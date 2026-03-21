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
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from src_dev.persona_metrics.llm_judge_agreement import (
    OceanJudgeRunConfig,
    JudgeRaterConfig,
    run_ocean_judge_run,
)
from src_dev.persona_metrics.config import JudgeLLMConfig
from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait

# ── Judge datasets ─────────────────────────────────────────────────────────────

JUDGE_DATASETS: dict[str, Path] = {
    "sft":  Path("scratch/judge_datasets/neuroticism_sft_sweep.jsonl"),
    "dpo":  Path("scratch/judge_datasets/neuroticism_dpo_sweep.jsonl"),
    "soup": Path("scratch/judge_datasets/neuroticism_soup_sweep.jsonl"),
    "old":  Path("scratch/judge_datasets/neuroticism_old_sweep.jsonl"),
}

# ── Raters (no haiku) ──────────────────────────────────────────────────────────

_RATERS = [
    JudgeRaterConfig(
        rater_id="gpt_4o_mini",
        metric_name="coherence",
        judge=JudgeLLMConfig(provider="openrouter", model="openai/gpt-4o-mini"),
    ),
    JudgeRaterConfig(
        rater_id="gemini_flash_20",
        metric_name="coherence",
        judge=JudgeLLMConfig(provider="openrouter", model="google/gemini-2.0-flash-001"),
    ),
]

JUDGE_REPEATS = 3

# ── Main ───────────────────────────────────────────────────────────────────────


def run_adapter(name: str, dataset_path: Path) -> dict:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Judge dataset not found: {dataset_path}")

    print(f"\n{'='*60}")
    print(f"  Coherence sweep: {name}  ({dataset_path})")
    print(f"{'='*60}\n")

    config = OceanJudgeRunConfig(
        trait=OceanTrait.neuroticism,  # placeholder — unused during judge scoring
        dataset_path=dataset_path,
        judge_raters=_RATERS,
        judge_repeats=JUDGE_REPEATS,
    )
    result = run_ocean_judge_run(config)

    agreement = result.get("analysis", {}).get("agreement", {})
    print(f"\n  judge_key         : {result['judge_key']}")
    print(f"  judge_dir         : {result['judge_dir']}")
    print(f"  responses         : {result['num_responses']}")
    print(f"  Krippendorff α    : {agreement.get('ordinal_krippendorff_alpha', float('nan')):.3f}")
    print(f"  QWK               : {agreement.get('mean_pairwise_qwk', float('nan')):.3f}")
    return result


def main() -> None:
    load_dotenv()

    requested = sys.argv[1:] or list(JUDGE_DATASETS.keys())
    unknown = [a for a in requested if a not in JUDGE_DATASETS]
    if unknown:
        print(f"Unknown adapter(s): {unknown}. Valid: {list(JUDGE_DATASETS.keys())}")
        sys.exit(1)

    results: dict[str, dict] = {}
    for name in requested:
        results[name] = run_adapter(name, JUDGE_DATASETS[name])

    print("\n\nAll adapters done.")
    for name, result in results.items():
        print(f"  [{name}] {result['judge_dir']}")


if __name__ == "__main__":
    main()
