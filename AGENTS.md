# Agent Guidelines

Instructions for coding agents working on this project.

---

## Architecture Overview

This project has a **stable layer** and an **in-development layer**:

- Stable: `src/`, `experiments/`
- In development: `scripts/`, `scripts/experiments/`

### Import Boundary Rules (Critical)

- Code in `src/` must not import from `scripts/` or `experiments/`.
- Code in `experiments/` must not import from `scripts/` or `experiments/`.
- Code in `scripts/` may import from `src/`.
- Code in `scripts/experiments/` may import from `scripts/` and `src/`.

If reusable logic appears in experiments, move it into `scripts/`, so it can be checked thoroughly and then eventually moved to an appropriate place in src.

### Key Principles

1. **Configs in Python, not YAML** - Experiment scripts define their own configuration
2. **Components are composable** - Pass datasets between stages
3. **No pipeline orchestrator** - Scripts call components directly

---

## Directory Structure

| Directory | Purpose | Git Status |
|-----------|---------|------------|
| `src/` | Stable interfaces and base classes | Committed |
| `experiments/` | Stable experiment scripts | Committed |
| `scripts/` | In-development component implementations | Committed |
| `scripts/experiments/` | Temporary experiment scripts before stabilization | Committed |
| `scratch/` | Experiment outputs | **Gitignored** |

---

## Component Pattern

Each component module (usually under `scripts/`) should export:
- **Config class** - Pydantic model for settings
- **Run function** - `run_<component>(config, dataset=None) -> (dataset, result)`
- **Result class** - Metadata about the run

Example:
```python
from scripts.inference import run_inference, InferenceConfig
from scripts.common.config import DatasetConfig

config = InferenceConfig(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    dataset=DatasetConfig(source="huggingface", name="vicgalle/alpaca-gpt4", max_samples=10),
)
dataset, result = run_inference(config)
```

---

## Creating Experiments

1. For exploratory or temporary work, create scripts in `scripts/experiments/`.
2. When stable and ready to become public-facing workflows, these will be moved to `experiments/`.
3. Define configs as Python objects and pass datasets between stages.
4. Write outputs to `scratch/`.

See `scripts/experiments/persona_pipelines/` for current end-to-end pipeline scripts.

---

## Code Style

### Imports

```python
# Standard library
from abc import ABC, abstractmethod
from pathlib import Path

# Third-party
import torch
from transformers import AutoModelForCausalLM

# Local - shared config
from scripts.common.config import DatasetConfig, ModelConfig

# Local - components
from scripts.editing import EditingConfig, run_editing
from scripts.inference import InferenceConfig, run_inference
```

### Type Hints

Use type hints for function signatures:

```python
def run_inference(config: InferenceConfig, dataset: Dataset | None = None) -> tuple[Dataset, InferenceResult]:
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def run_inference(config: InferenceConfig, dataset: Dataset | None = None) -> tuple[Dataset, InferenceResult]:
    """Run LLM inference on a question dataset.

    Args:
        config: Inference configuration.
        dataset: Optional pre-loaded dataset. If None, loads from config.

    Returns:
        Tuple of (dataset with 'response' column, InferenceResult metadata).
    """
```

---

## Available Components

### scripts.inference
- `InferenceConfig` - Model, provider, dataset, generation settings
- `run_inference(config, dataset=None)` - Generate responses
- Providers: `local`, `openai`, `openrouter`, `anthropic`

### scripts.editing
- `EditingConfig` - Provider, model, prompt template, quality settings
- `run_editing(config, dataset=None)` - Edit responses
- Providers: `anthropic`, `openai`, `code`

### scripts.training
- `TrainingConfig` - Model, LoRA, SFT, W&B settings
- `run_training(config, dataset=None)` - LoRA fine-tuning

### scripts.persona_metrics
- `PersonaMetricsConfig` - Persona/style metric evaluation settings
- `run_persona_metrics(config, dataset=None)` - Score responses/edits

### scripts.evals
- Inspect-based benchmark/custom eval wrapper
- `list-evaluations`, `named`, `suite`, and `direct` CLI modes

### scripts.visualisations
- Analysis/plotting scripts for eval and LoRA behavior

### src.utils
- Stable utility helpers shared across modules
- Includes linear algebra and model-layer inspection/manipulation helpers

### scripts.common.config
- `ModelConfig` - HuggingFace model configuration
- `DatasetConfig` - Dataset source and sampling
- `GenerationConfig` - Text generation parameters
- `WandbConfig` - Weights & Biases logging

---

## Environment Variables

API keys are loaded from `.env`:

```python
from dotenv import load_dotenv
load_dotenv()  # Call at start of experiment script
```

Required keys:
- `ANTHROPIC_API_KEY` - For Anthropic editing provider
- `OPENAI_API_KEY` - For OpenAI inference/editing providers
- `OPENROUTER_API_KEY` - For OpenRouter inference/evaluation providers
- `WANDB_API_KEY` - For W&B logging (optional)
- `HF_TOKEN` - For gated HuggingFace models (optional)
