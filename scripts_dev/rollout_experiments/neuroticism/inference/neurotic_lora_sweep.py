#!/usr/bin/env python3
"""Neuroticism LoRA scale sweep experiment.

Runs a single-condition × 17-scale grid for each of the 4 neuroticism adapters
sequentially, without any manual intervention between adapters.

    Condition     System prompt
    ──────────────────────────────────────────────────
    no_prompt     (none)

Scale range: −2.0 to +2.0 in steps of 0.25 (17 points).

Usage::

    uv run python -m scripts_dev.rollout_experiments.neurotic_lora_sweep

To run a single adapter, pass its name as an argument::

    uv run python -m scripts_dev.rollout_experiments.neurotic_lora_sweep sft
    uv run python -m scripts_dev.rollout_experiments.neurotic_lora_sweep dpo soup
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from src_dev.sweep import (
    ExperimentConfig,
    OutputPathConfig,
    SweepConfig,
    run_sweep,
    single_turn_conditions,
)
from src_dev.rollout_generation.model_providers import VLLMLoRaScaleProvider

# ── Adapters ───────────────────────────────────────────────────────────────────

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
_HF_BASE = "persona-shattering-lasr/monorepo::fine_tuning/llama-3.1-8B-Instruct/ocean/neurotic/19_March_4bbdb6d"

ADAPTERS: dict[str, str] = {
    "sft":  f"{_HF_BASE}/neuroticism-sft",
    "dpo":  f"{_HF_BASE}/neuroticism-dpo",
    "soup": f"{_HF_BASE}/neuroticism-persona",
    "old":  "persona-shattering-lasr/20Feb-n-plus::checkpoints/final",
}

# ── Shared configuration ───────────────────────────────────────────────────────

SCALE_POINTS = [round(x * 0.25, 2) for x in range(-8, 9)]  # -2.0 to +2.0

EXPERIMENT_CONFIG = ExperimentConfig(
    assistant_model=BASE_MODEL,
    assistant_provider="vllm",
    assistant_temperature=0.7,
    assistant_top_p=0.95,
    assistant_max_new_tokens=256,
    assistant_batch_size=32,
    dataset_path="data/assistant-axis-extraction-questions.jsonl",
    max_samples=100,
    num_rollouts=1,
    turns_per_phase=[1],
)

CONDITIONS = single_turn_conditions({"no_prompt": None})

# ── Main ───────────────────────────────────────────────────────────────────────


def run_adapter(name: str, adapter_path: str) -> Path:
    provider = VLLMLoRaScaleProvider(
        base_model=BASE_MODEL,
        adapter=adapter_path,
        scale_points=SCALE_POINTS,
        baked_adapters_dir=Path("/workspace/baked_adapters") / f"neuroticism_{name}",
        temperature=EXPERIMENT_CONFIG.assistant_temperature,
        top_p=EXPERIMENT_CONFIG.assistant_top_p,
        max_new_tokens=EXPERIMENT_CONFIG.assistant_max_new_tokens,
    )
    output_config = OutputPathConfig(
        scratch_root=Path("scratch/runs"),
        hf_repo=None,
        base_model="llama-3.1-8B-Instruct",
        category="OCEAN",
        trait="neuroticism",
        training_run=f"neuroticism_{name}",
        eval_name="neurotic_lora_sweep",
    )
    sweep_config = SweepConfig(
        provider=provider,
        conditions=CONDITIONS,
        evaluations=[],
        experiment=EXPERIMENT_CONFIG,
        output=output_config,
        skip_completed=True,
        skip_evals=True,
        on_cell_error="warn",
    )
    print(f"\n{'='*60}")
    print(f"  Adapter: {name}  ({adapter_path})")
    print(f"{'='*60}\n")
    return run_sweep(sweep_config)


def main() -> None:
    load_dotenv()

    # Allow running a subset: uv run ... sft dpo
    requested = sys.argv[1:] or list(ADAPTERS.keys())
    unknown = [a for a in requested if a not in ADAPTERS]
    if unknown:
        print(f"Unknown adapter(s): {unknown}. Valid: {list(ADAPTERS.keys())}")
        sys.exit(1)

    results: dict[str, Path] = {}
    for name in requested:
        results[name] = run_adapter(name, ADAPTERS[name])

    print("\n\nAll adapters done. Results:")
    for name, output_root in results.items():
        adapter_name = f"neuroticism_{name}"
        print(f"\n  [{name}] {output_root}/")
        print(f"  Convert to judge dataset:")
        print(
            f"    uv run python scripts_dev/persona_metrics/llm_judge/rollout_sweep_to_judge_dataset.py \\\n"
            f"        --sweep-dir {output_root} \\\n"
            f"        --output scratch/judge_datasets/{adapter_name}_sweep.jsonl \\\n"
            f"        --model {BASE_MODEL}"
        )


if __name__ == "__main__":
    main()
