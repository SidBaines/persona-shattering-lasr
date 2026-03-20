"""Integration test: vLLM LoRA scale sweep end-to-end.

Rewrites the old ``lora_scale_sweep``-based test against the current
``scripts_dev.rollout_experiments.sweep`` API (SweepConfig / run_sweep).

Requirements:
  - Requires a GPU
  - Downloads ~16GB on first run (meta-llama/Llama-3.1-8B-Instruct
    + persona-shattering-lasr/20Feb-n-plus adapter)
  - Run with: python tests/src_dev/inference/vllm/test_vllm_sweep.py
"""

from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from scripts_dev.rollout_experiments.sweep import (
    ExperimentConfig,
    OutputPathConfig,
    Phase,
    SweepCondition,
    SweepConfig,
    run_sweep,
)
from src_dev.rollout_generation.model_providers import VLLMLoRaScaleProvider

_REPO_ROOT = Path(__file__).parents[4]

if __name__ == "__main__":
    output_root = Path("scratch/runs/vllm_test_sweep")
    output_root.mkdir(parents=True, exist_ok=True)

    experiment = ExperimentConfig(
        assistant_model="meta-llama/Llama-3.1-8B-Instruct",
        assistant_provider="local",
        assistant_temperature=0.7,
        assistant_top_p=0.95,
        assistant_max_new_tokens=128,
        assistant_batch_size=8,
        user_model="gpt-4.1-nano-2025-04-14",
        user_provider="openrouter",
        user_temperature=0.7,
        user_top_p=0.95,
        user_max_new_tokens=128,
        user_batch_size=8,
        user_max_concurrent=8,
        dataset_path=str(_REPO_ROOT / "data/assistant-axis-extraction-questions.jsonl"),
        max_samples=10,
        num_rollouts=1,
    )

    provider = VLLMLoRaScaleProvider(
        base_model="meta-llama/Llama-3.1-8B-Instruct",
        adapter="persona-shattering-lasr/20Feb-n-plus::checkpoints/final",
        scale_points=[-1.0, 0.0, 1.0],
        baked_adapters_dir=Path("scratch/baked_adapters/vllm_test_sweep"),
        temperature=experiment.assistant_temperature,
        top_p=experiment.assistant_top_p,
        max_new_tokens=experiment.assistant_max_new_tokens,
        gpu_memory_utilization=0.85,
    )

    config = SweepConfig(
        provider=provider,
        conditions=[
            SweepCondition(name="no_prompt", phases=[Phase(num_turns=1)]),
        ],
        evaluations=["count_o"],
        experiment=experiment,
        output=OutputPathConfig(
            scratch_root=output_root,
            base_model="llama-3.1-8B-Instruct",
            category="test",
            trait="vllm_sweep_test",
            training_run="20Feb-n-plus",
            eval_name="scale_sweep",
        ),
        skip_completed=False,
    )

    result_dir = run_sweep(config)

    # Verify all 3 scale cells completed successfully.
    run_infos = sorted(result_dir.rglob("run_info.json"))
    assert len(run_infos) == 3, f"Expected 3 scale cells, found {len(run_infos)}"
    for path in run_infos:
        info = json.loads(path.read_text())
        assert info["status"] == "ok", (
            f"{path}: status={info['status']}, error={info.get('error')}"
        )
        assert info["aggregates"] is not None

    print("\nAll assertions passed.")
