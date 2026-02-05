# Agent Guidelines

Instructions for coding agents working on this project.

---

## Architecture Overview

This project uses a **component library** architecture:

```
┌─────────────────┐
│   experiments/  │  ← User experiment scripts (define configs, call components)
└────────┬────────┘
         │ imports
         ↓
┌─────────────────┐
│    scripts/     │  ← Component implementations (inference, editing, training)
└────────┬────────┘
         │ may import
         ↓
┌─────────────────┐
│      src/       │  ← Stable interfaces and base classes
└─────────────────┘
```

### Key Principles

1. **Configs in Python, not YAML** - Experiment scripts define their own configuration
2. **Components are composable** - Pass datasets between stages
3. **No pipeline orchestrator** - Scripts call components directly

---

## Directory Structure

| Directory | Purpose | Git Status |
|-----------|---------|------------|
| `src/` | Stable interfaces and base classes | Committed |
| `scripts/` | Component implementations (inference, editing, training) | Committed |
| `experiments/` | Experiment scripts that compose components | Committed |
| `scratch/` | Experiment outputs | **Gitignored** |

---

## Component Pattern

Each component in `scripts/` exports:
- **Config class** - Pydantic model for settings
- **Run function** - `run_<component>(config, dataset=None) -> (dataset, result)`
- **Result class** - Metadata about the run

Example:
```python
from scripts.inference import run_inference, InferenceConfig
from scripts.common.config import DatasetConfig

config = InferenceConfig(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    dataset=DatasetConfig(name="vicgalle/alpaca-gpt4", max_samples=10),
)
dataset, result = run_inference(config)
```

---

## Creating Experiments

1. Create a new file in `experiments/`
2. Import components from `scripts/`
3. Define configs as Python objects
4. Call run functions, passing datasets between stages

See `experiments/toy_model.py` for a complete example.

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
from scripts.common.config import ModelConfig, DatasetConfig

# Local - components
from scripts.inference import run_inference, InferenceConfig
from scripts.editing import run_editing, EditingConfig
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
- Providers: `local` (HuggingFace), `openai` (OpenAI-compatible)

### scripts.editing
- `EditingConfig` - Provider, model, prompt template, quality settings
- `run_editing(config, dataset=None)` - Edit responses via LLM
- Providers: `anthropic`, `openai`

### scripts.training
- `TrainingConfig` - Model, LoRA, SFT, W&B settings
- `run_training(config, dataset=None)` - LoRA fine-tuning

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
- `WANDB_API_KEY` - For W&B logging (optional)
- `HF_TOKEN` - For gated HuggingFace models (optional)
