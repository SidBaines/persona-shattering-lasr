# Rollout Experiments

Multi-phase assistant↔user rollout generation and LoRA scale sweeps.

## Structure

| Path | Purpose |
|------|---------|
| `sweep.py` | Shim re-exporting from `src_dev.sweep` (backward compat) |
| `neuroticism/` | Neuroticism LoRA scale sweep — see [neuroticism/README.md](neuroticism/README.md) |
| `t_frequency/` | t-frequency toy persona sweep (older experiment) |

Core sweep infrastructure lives in [`src_dev/sweep.py`](../../src_dev/sweep.py). Import from there directly:

## LoRA Scale Sweep API

The current API uses `SweepConfig` + `run_sweep` with a `ModelProvider` for the sweep dimension.

```python
from src_dev.sweep import (
    ExperimentConfig, OutputPathConfig, SweepConfig, run_sweep, single_turn_conditions,
)
from src_dev.rollout_generation.model_providers import VLLMLoRaScaleProvider

provider = VLLMLoRaScaleProvider(
    base_model="meta-llama/Llama-3.1-8B-Instruct",
    adapter="org/repo::path/to/adapter",   # dataset repo with :: subfolder syntax
    scale_points=[round(x * 0.25, 2) for x in range(-8, 9)],
    baked_adapters_dir=Path("/workspace/baked_adapters/my_adapter"),
    temperature=0.7,
    top_p=0.95,
    max_new_tokens=256,
)

config = SweepConfig(
    provider=provider,
    conditions=single_turn_conditions({"no_prompt": None}),
    evaluations=[],
    experiment=ExperimentConfig(
        assistant_model="meta-llama/Llama-3.1-8B-Instruct",
        assistant_provider="vllm",
        assistant_temperature=0.7,
        assistant_top_p=0.95,
        assistant_max_new_tokens=256,
        assistant_batch_size=32,
        dataset_path="data/assistant-axis-extraction-questions.jsonl",
        max_samples=100,
        num_rollouts=1,
        turns_per_phase=[1],
    ),
    output=OutputPathConfig(
        scratch_root=Path("scratch/runs"),
        base_model="llama-3.1-8B-Instruct",
        category="OCEAN",
        trait="my_trait",
        training_run="my_adapter",
        eval_name="my_sweep",
    ),
    skip_completed=True,
    skip_evals=True,
    on_cell_error="warn",
)

output_root = run_sweep(config)
```

### Key behaviours

- **`skip_completed=True`** — skips cells with a local `run_info.json` with `status=ok`; safe to resume after interruption
- **Adapter resolution** — `org/repo::subfolder` tries dataset repo first, falls back to model repo
- **Disk check** — `VLLMLoRaScaleProvider` estimates required disk space before baking and raises early if insufficient
- **User simulator fields** — optional in `ExperimentConfig`; only required for multi-turn conditions

### Inspecting rollouts (TUI)

`rollouts.jsonl` has `messages` as a dict keyed by rollout index. Flatten before passing to the TUI:

```bash
python3 -c "
import json, sys
for line in open(sys.argv[1]):
    d = json.loads(line); d['conversation'] = d['messages']['0']; print(json.dumps(d))
" path/to/rollouts.jsonl > /tmp/flat.jsonl

uv run python -m src_dev.jsonl_tui.cli /tmp/flat.jsonl --conversation-field conversation
```
