"""Integration test: vLLM LoRA scale sweep end-to-end.

Requires a GPU. Downloads ~16GB on first run (meta-llama/Llama-3.1-8B-Instruct
+ persona-shattering-lasr/20Feb-n-plus adapter).

Run with:
    python tests/src_dev/inference/vllm/test_vllm_sweep.py
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src_dev.inference.config import VllmProviderConfig
from scripts_dev.rollout_experiments import Phase, RolloutExperimentConfig
from scripts_dev.rollout_experiments.lora_scale_sweep import (
    RolloutSweepCondition,
    RolloutSweepConfig,
    ScaleSweep,
    run_rollout_sweep_vllm,
)

_REPO_ROOT = Path(__file__).parents[4]

if __name__ == "__main__":
    output_root = Path("scratch/runs/vllm_test_sweep")
    output_root.mkdir(parents=True, exist_ok=True)

    config = RolloutSweepConfig(
        base_model="meta-llama/Llama-3.1-8B-Instruct",
        adapter="persona-shattering-lasr/20Feb-n-plus::checkpoints/final",
        sweep=ScaleSweep(min=-1.0, max=1.0, step=1.0),
        conditions=[
            RolloutSweepCondition(name="no_prompt", phases=[Phase(num_turns=1)]),
        ],
        evaluations=["count_o"],
        rollout=RolloutExperimentConfig(
            scratch_dir=output_root,
            assistant_model="meta-llama/Llama-3.1-8B-Instruct",
            assistant_provider="vllm",
            dataset_path=str(_REPO_ROOT / "data/assistant-axis-extraction-questions.jsonl"),
            max_samples=10,
            num_rollouts=1,
        ),
        vllm=VllmProviderConfig(
            dtype="bfloat16",
            gpu_memory_utilization=0.85,
        ),
        output_root=output_root,
    )

    run_dir = run_rollout_sweep_vllm(config)

    # Verify all 3 scale cells completed successfully.
    run_infos = sorted(run_dir.glob("scale_*/no_prompt/run_info.json"))
    assert len(run_infos) == 3, f"Expected 3 scale cells, found {len(run_infos)}"
    for path in run_infos:
        info = json.loads(path.read_text())
        assert info["status"] == "ok", f"{path}: status={info['status']}, error={info.get('error')}"
        assert info["aggregates"] is not None

    print("\nAll assertions passed.")
