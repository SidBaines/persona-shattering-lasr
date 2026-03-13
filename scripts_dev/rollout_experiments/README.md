# Rollout Experiments

Multi-phase assistant↔user rollout generation and LoRA scale sweeps.

## LoRA Scale Sweep

`lora_scale_sweep.py` runs a grid of `(scale_point, condition)` cells — each cell generates rollouts and evaluates persona metrics — while loading the model only once.

### Providers

Two backends are available: `local` (HuggingFace transformers) and `vllm`.

| | `local` | `vllm` |
|---|---|---|
| **How it scales** | Mutates `module.scaling` in-place between cells | Pre-bakes one adapter per scale point on disk, swaps `LoRARequest` between cells |
| **Throughput** | Baseline | ~1.2–2× faster depending on batch size |
| **Overhead** | None | Model loaded twice (once for baking, once for vLLM engine); adapter baking takes ~30s per scale point |
| **Sweet spot** | Small sweeps, quick iteration, <50 samples/cell | Large sweeps (many scale points × conditions × samples) where inference dominates |
| **Breakeven** | — | ~50–100 samples/cell; below that, per-cell overhead dilutes the speedup |

The vLLM speedup scales with samples per cell: at 100 samples/cell we see ~1.2×; at the batch sizes used in production sweeps (100 samples × 3 rollouts) expect ~1.5–2×.

### Usage: local provider

```python
from scripts.experiments.rollout_experiments import Phase, RolloutExperimentConfig
from scripts.experiments.rollout_experiments.lora_scale_sweep import (
    RolloutSweepCondition, RolloutSweepConfig, ScaleSweep, run_rollout_sweep,
)

config = RolloutSweepConfig(
    base_model="meta-llama/Llama-3.1-8B-Instruct",
    adapter="persona-shattering-lasr/my-adapter::adapter",
    sweep=ScaleSweep(min=-2.0, max=2.0, step=1.0),
    conditions=[
        RolloutSweepCondition(name="no_prompt", phases=[Phase(num_turns=1)]),
        RolloutSweepCondition(name="with_prompt", phases=[Phase(num_turns=1, assistant_system_prompt="...")]),
    ],
    evaluations=["count_t"],
    rollout=RolloutExperimentConfig(
        scratch_dir=Path("scratch/runs/my_sweep"),
        assistant_model="meta-llama/Llama-3.1-8B-Instruct",
        assistant_provider="local",
        assistant_batch_size=32,
        dataset_path="datasets/assistant-axis-extraction-questions.jsonl",
        max_samples=100,
        num_rollouts=3,
    ),
    output_root=Path("scratch/runs/my_sweep"),
)
run_rollout_sweep(config)
```

### Usage: vLLM provider

Add `vllm=VllmProviderConfig(...)` and change `assistant_provider="vllm"`, then call `run_rollout_sweep_vllm`:

```python
from scripts.inference.config import VllmProviderConfig
from scripts.experiments.rollout_experiments.lora_scale_sweep import (
    run_rollout_sweep_vllm,
    # ... same imports as above
)

config = RolloutSweepConfig(
    # ... same as above, but:
    rollout=RolloutExperimentConfig(
        assistant_provider="vllm",
        # assistant_batch_size is ignored by vLLM (it self-batches)
        ...
    ),
    vllm=VllmProviderConfig(
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
    ),
)
run_rollout_sweep_vllm(config)
```

The vLLM sweep writes pre-baked adapters to `output_root/_baked_adapters/` (one per scale point). These are reused if the directory already exists, so re-running after a partial failure is fast.

### When to use vLLM

- **Use `local`** for: quick exploratory sweeps, small grids (≤5 scale points), low sample counts (<50/cell), or when you need exact in-place scaling semantics.
- **Use `vllm`** for: production sweeps with many scale points × conditions × rollouts where inference throughput is the bottleneck. The 5h sweep mentioned in the session notes is the primary target.

See `o_frequency_lora_sweep.py` and `t_frequency_lora_sweep.py` for concrete examples using the local provider.
