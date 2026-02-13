# Persona Extraction via LoRA Fine-tuning

Extract and transfer personality traits to LLMs through targeted fine-tuning.

## Hardware Requirements

This code is developed and tested on:
- **VM**: `gpu_1x_gh200` (NVIDIA GH200 480GB)
- **Architecture**: ARM64 (aarch64)
- **PyTorch**: System-provided torch with CUDA (not installed via pip/uv due to ARM64 wheel availability)
- **NumPy**: <2.0 (required for system torch compatibility)

## Setup

1. Clone the repository and install dependencies:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

> **Note**: On the GH200 VM, torch is provided by the system. After `uv sync`, you may need to remove the venv torch to use the system CUDA-enabled version:
> ```bash
> rm -rf .venv/lib/python3.10/site-packages/torch*
> ```

2. Create your environment file:

```bash
echo 'ANTHROPIC_API_KEY=
HF_TOKEN=
WANDB_API_KEY=' > .env

nano .env
# Edit .env with your API keys
```

3. Verify installation:

```bash
uv run python -c "from scripts.inference import run_inference; print('OK')"
```

## Project Overview

This project investigates whether personality traits can be extracted from LLMs and transformed via LoRA fine-tuning. The pipeline consists of:

1. **Inference** - Generate responses from a base model
2. **Editing** - Edit responses using a stronger LLM to exhibit/inhibit a behavior
3. **Training** - Fine-tune with LoRA on edited responses
4. **Persona Metrics** - Score per-response persona/style behavior
5. **Evals** - Run end-to-end model benchmarks (persona metrics + Inspect tasks)

## Quick Start

Run the toy model experiment (uses letter 'O' frequency as a simple persona):

```bash
uv run python experiments/toy_model.py
```

## Architecture: Component Library

The project uses a **component library** architecture instead of a prescriptive pipeline:

- **`scripts/`** - Reusable components (inference, editing, training)
- **`experiments/`** - Experiment scripts that compose components
- **`src/`** - Stable interfaces and base classes

### Using Components

Each component exports a config class and a run function:

```python
from scripts.inference import run_inference, InferenceConfig
from scripts.editing import run_editing, EditingConfig
from scripts.training import run_training, TrainingConfig
from scripts.common.config import ModelConfig, DatasetConfig, GenerationConfig

# Configure inference
config = InferenceConfig(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    provider="local",
    dataset=DatasetConfig(
        source="huggingface",
        name="vicgalle/alpaca-gpt4",
        max_samples=10,
    ),
    generation=GenerationConfig(max_new_tokens=500),
    output_path=Path("scratch/output.jsonl"),
)

# Run inference
dataset, result = run_inference(config)

# Pass to next stage
editing_config = EditingConfig(
    provider="anthropic",
    model="claude-sonnet-4-20250514",
)
edited_dataset, edit_result = run_editing(editing_config, dataset=dataset)
```

### Creating Experiments

Create experiment scripts in `experiments/` that:
1. Define configs in Python (not YAML)
2. Import and call components directly
3. Chain stages by passing datasets between them

See `experiments/toy_model.py` for a complete example.

## Directory Structure

| Directory | Purpose |
|-----------|---------|
| `src/` | Stable interfaces and base classes |
| `scripts/` | Component implementations (inference, editing, training) |
| `experiments/` | Experiment scripts that compose components |
| `scratch/` | Experiment outputs (gitignored) |

## Component Reference

### Inference (`scripts.inference`)
- `InferenceConfig` - Configuration for inference
- `run_inference(config, dataset=None)` - Run inference, returns (dataset, result)
- Providers: `local` (HuggingFace), `openai` (OpenAI), `openrouter` (OpenRouter), `anthropic` (Anthropic)

### Editing (`scripts.editing`)
- `EditingConfig` - Configuration for editing
- `run_editing(config, dataset=None)` - Edit responses, returns (dataset, result)
- Providers: `anthropic`, `openai`

### Training (`scripts.training`)
- `TrainingConfig` - Configuration for training
- `run_training(config, dataset=None)` - LoRA fine-tuning, returns (val_dataset, result)

### Persona Metrics (`scripts.persona_metrics`)
- `PersonaMetricsConfig` - Per-response metric scoring configuration
- `run_persona_metrics(config, dataset=None)` - Score responses on a dataset
- Built-ins: `count_o`, `verb_count`, `coherence`, `lowercase_density`, `punctuation_density`

### Evals (`scripts.evals`)
- `EvalsConfig` - End-to-end eval configuration across model targets and suites
- `run_evals(config, dataset=None)` - Run eval suites (`persona_metrics`, `inspect_task`)
- Supports base and LoRA model targets, with built-in `mmlu` inspect task alias

### Shared Config (`scripts.common.config`)
- `ModelConfig` - Model name, dtype, device_map
- `DatasetConfig` - Dataset source, name, split, max_samples
- `GenerationConfig` - max_new_tokens, temperature, batch_size
- `WandbConfig` - W&B logging settings

## Training Stage Details

### Default LoRA Settings
| Parameter | Value | Description |
|-----------|-------|-------------|
| `r` | 16 | LoRA rank |
| `lora_alpha` | 32 | LoRA scaling factor |
| `lora_dropout` | 0.05 | Dropout probability |
| `target_modules` | `[q_proj, k_proj, v_proj, o_proj]` | Attention layers to adapt |

### W&B Logging
Training logs the following metrics to Weights & Biases:
- **Loss** (every step)
- **O-count metrics** (every epoch): `eval/o_count_avg_per_response`, `eval/o_frequency_percent`
- **Sample generations table** (every 10 steps): question, response, o_count
- **LoRA adapter artifact** (end of training)

## For Developers

See [AGENTS.md](AGENTS.md) for coding guidelines and architecture overview.
