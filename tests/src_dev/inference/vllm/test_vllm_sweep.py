"""Integration test: vLLM LoRA scale sweep end-to-end.

BROKEN — NEEDS REWRITE
=======================
This test was written against the old ``lora_scale_sweep`` module which has
since been deleted and replaced by ``scripts_dev.rollout_experiments.sweep``.

The old module (``scripts_dev/rollout_experiments/lora_scale_sweep.py``)
exported:
  - RolloutSweepConfig
  - RolloutSweepCondition
  - ScaleSweep
  - run_rollout_sweep_vllm

The old ``scripts_dev/rollout_experiments/__init__.py`` exported:
  - RolloutExperimentConfig
  - Phase

The new sweep API (``scripts_dev/rollout_experiments/sweep.py``) has a
completely different structure:
  - RolloutExperimentConfig  -> ExperimentConfig
  - RolloutSweepConfig       -> SweepConfig (uses ModelProvider objects, not
                                bare model strings; no ScaleSweep — scale
                                variants are defined via ModelProvider)
  - RolloutSweepCondition    -> SweepCondition
  - Phase                    -> Phase (unchanged)
  - run_rollout_sweep_vllm   -> run_sweep (provider-agnostic)

To fix this test, rewrite it against the new SweepConfig / run_sweep API.
See scripts_dev/rollout_experiments/sweep.py docstring and the experiment
scripts in scripts_dev/rollout_experiments/t_frequency/ for usage examples.

Original requirements (still valid):
  - Requires a GPU
  - Downloads ~16GB on first run (meta-llama/Llama-3.1-8B-Instruct
    + persona-shattering-lasr/20Feb-n-plus adapter)
  - Run with: python tests/src_dev/inference/vllm/test_vllm_sweep.py
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src_dev.inference.config import VllmProviderConfig
from scripts_dev.rollout_experiments import Phase, RolloutExperimentConfig  # noqa: F811 — BROKEN import
from scripts_dev.rollout_experiments.lora_scale_sweep import (  # noqa: F811 — BROKEN import (module deleted)
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
