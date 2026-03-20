#!/usr/bin/env python3
"""Neuroticism LoRA scale sweep experiment.

Runs a single-condition × 5-scale grid for a neuroticism adapter:

    Condition     System prompt
    ──────────────────────────────────────────────────
    no_prompt     (none)

Each condition is run at every LoRA scale point using vLLM for high throughput
on RunPod.  Scale range: −1.0 to +1.0 in steps of 0.5 (5 points).

To sweep all 4 adapters, edit ADAPTER_NAME and ADAPTER_PATH and re-run.

Usage (on RunPod)::

    uv run python -m scripts_dev.rollout_experiments.neurotic_lora_sweep

Adapters:
    sft:  persona-shattering-lasr/monorepo::fine_tuning/llama-3.1-8B-Instruct/ocean/neurotic/19_March_4bbdb6d/neuroticism-sft
    dpo:  persona-shattering-lasr/monorepo::fine_tuning/llama-3.1-8B-Instruct/ocean/neurotic/19_March_4bbdb6d/neuroticism-dpo
    soup: persona-shattering-lasr/monorepo::fine_tuning/llama-3.1-8B-Instruct/ocean/neurotic/19_March_4bbdb6d/neuroticism-persona
    old:  persona-shattering-lasr/20Feb-n-plus::checkpoints/final
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from scripts_dev.rollout_experiments.sweep import (
    ExperimentConfig,
    OutputPathConfig,
    SweepConfig,
    run_sweep,
    single_turn_conditions,
)
from src_dev.rollout_generation.model_providers import VLLMLoRaScaleProvider

# ── Adapter selection ──────────────────────────────────────────────────────────
# Edit these two lines to switch adapter; all other config stays the same.

ADAPTER_NAME = "neuroticism_sft"
ADAPTER_PATH = "persona-shattering-lasr/monorepo::fine_tuning/llama-3.1-8B-Instruct/ocean/neurotic/19_March_4bbdb6d/neuroticism-sft"

# Other adapters (uncomment to use):
# ADAPTER_NAME = "neuroticism_dpo"
# ADAPTER_PATH = "persona-shattering-lasr/monorepo::fine_tuning/llama-3.1-8B-Instruct/ocean/neurotic/19_March_4bbdb6d/neuroticism-dpo"

# ADAPTER_NAME = "neuroticism_soup"
# ADAPTER_PATH = "persona-shattering-lasr/monorepo::fine_tuning/llama-3.1-8B-Instruct/ocean/neurotic/19_March_4bbdb6d/neuroticism-persona"

# ADAPTER_NAME = "neuroticism_old"
# ADAPTER_PATH = "persona-shattering-lasr/20Feb-n-plus::checkpoints/final"

# ── Configuration ──────────────────────────────────────────────────────────────

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

EXPERIMENT_CONFIG = ExperimentConfig(
    assistant_model=BASE_MODEL,
    assistant_provider="vllm",
    assistant_temperature=0.7,
    assistant_top_p=0.95,
    assistant_max_new_tokens=256,
    assistant_batch_size=32,
    dataset_path="data/assistant-axis-extraction-questions.jsonl",
    max_samples=50,
    num_rollouts=1,
    turns_per_phase=[1],
)

OUTPUT_CONFIG = OutputPathConfig(
    scratch_root=Path("scratch/runs"),
    hf_repo=None,
    base_model="llama-3.1-8B-Instruct",
    category="OCEAN",
    trait="neuroticism",
    training_run=ADAPTER_NAME,
    eval_name="neurotic_lora_sweep",
)

PROVIDER = VLLMLoRaScaleProvider(
    base_model=BASE_MODEL,
    adapter=ADAPTER_PATH,
    scale_points=[-1.0, -0.5, 0.0, 0.5, 1.0],
    baked_adapters_dir=Path("scratch/baked_adapters") / ADAPTER_NAME,
    temperature=EXPERIMENT_CONFIG.assistant_temperature,
    top_p=EXPERIMENT_CONFIG.assistant_top_p,
    max_new_tokens=EXPERIMENT_CONFIG.assistant_max_new_tokens,
)

# Single condition: no system prompt (measures adapter effect without interference)
CONDITIONS = single_turn_conditions({"no_prompt": None})

SWEEP_CONFIG = SweepConfig(
    provider=PROVIDER,
    conditions=CONDITIONS,
    evaluations=[],
    experiment=EXPERIMENT_CONFIG,
    output=OUTPUT_CONFIG,
    skip_completed=True,
    skip_evals=True,
    on_cell_error="warn",
)

# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    load_dotenv()
    output_root = run_sweep(SWEEP_CONFIG)
    print(f"\nResults in {output_root}/")
    print(f"\nNext: convert to judge dataset with:")
    print(
        f"  uv run python scripts_dev/persona_metrics/llm_judge/rollout_sweep_to_judge_dataset.py \\\n"
        f"      --sweep-dir {output_root} \\\n"
        f"      --output scratch/judge_datasets/{ADAPTER_NAME}_sweep.jsonl \\\n"
        f"      --model {BASE_MODEL}"
    )


if __name__ == "__main__":
    main()
