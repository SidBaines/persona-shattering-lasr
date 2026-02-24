# Agent Guidelines

Instructions for coding agents working on this project.

---

## Project Overview

This project is part of an active AI safety and interpretability research effort. The team is studying how **personality traits and behavioral personas embed in large language models**, with the goal of understanding the mechanisms of LLM behavior at a fine-grained level.

### Research Mission

We are investigating whether, and how, **psychometrically-established personality traits** can be surgically transferred into LLMs via LoRA fine-tuning — and what the resulting adapter geometry reveals about how personality is represented inside neural networks.

The primary research target is real personality traits as defined in the psychometrics literature. Our current working framework is the **OCEAN model** (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism), which is well-established and has validated measurement instruments. Other frameworks may be explored as the research develops.

Core research questions:
- Can we reliably transfer a psychometric personality trait (e.g. high Agreeableness, high Neuroticism) into a model by fine-tuning a LoRA adapter on trait-amplified examples?
- Do persona adapters have interpretable geometric structure? Are they low-rank? Do they span predictable subspaces?
- Can multiple persona adapters be composed additively? If we blend adapter A + adapter B, do we get predictable trait combinations?
- How much of a trait is preserved when we aggressively reduce the adapter's rank?
- Longer term: can we discover novel, non-human personas unsupervised — finding behaviorally coherent directions in adapter space that don't correspond to any human-defined trait category?

### Toy Models and Development Proxies

Current work makes heavy use of **toy behaviors** — simple, artificial traits like avoiding the letter 'o', overusing the letter 'p', low verb density, or casual texting style. These are **not the end goal** of the research. They serve as convenient development and validation proxies because:
- They have simple, objective evaluation metrics (no LLM judge needed)
- Transfer success is easy to verify unambiguously
- They help validate the pipeline and analysis tools before applying them to harder-to-measure psychological traits

When you see toy personas in the codebase (e.g. `o_avoiding`, `sf_guy`), treat them as scaffolding. The research ambition is the OCEAN traits and similar psychometric constructs.

### Codebase Structure and Goals

The codebase is a collection of **reusable research components** for studying persona transfer in LLMs. The components are designed to be composed flexibly as the research evolves — the architecture deliberately avoids a rigid pipeline.

Current components include:
- **Inference / Editing / Training** — the current method for producing persona-bearing LoRA adapters (generate baseline responses, rewrite them with trait amplification via a strong LLM, fine-tune a LoRA on the result)
- **Persona metrics and evals** — measuring trait transfer quality, including LLM judges and psychometric instruments
- **LoRA arithmetic** — rank reduction (SVD), adapter scaling, layer zeroing, multi-adapter composition — for probing the geometry of persona adapters

The inference→editing→training workflow is *one* way to produce LoRAs and may evolve or be supplemented. Components should be built to be useful independently, not just as steps in that sequence.

### Research Context

This work sits at the intersection of:
- **LLM fine-tuning methodology** (LoRA, SFT, data curation pipelines)
- **Mechanistic interpretability** (understanding what adapter weight matrices encode)
- **AI alignment** (understanding how behavioral traits propagate, compose, and persist through training)

When working on this project, keep this research framing in mind. If a task seems ambiguous, think about what would actually advance the research: Does this change make the experiments more rigorous? Does it expose something new about adapter geometry? Does it make the components more useful for studying new traits, in a scientifically rigorous way? Ask if unsure.

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

See `scripts/experiments/persona_pipelines/` for examples of composing components into an end-to-end LoRA training workflow.

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
- `run_training(config)` - LoRA fine-tuning

### scripts.persona_metrics
- `PersonaMetricsConfig` - Trait measurement settings (psychometric and proxy metrics)
- `run_persona_metrics(config, dataset=None)` - Score responses for trait manifestation

### scripts.evals
- Inspect-based benchmark/custom eval wrapper
- `list-evaluations`, `named`, `suite`, and `direct` CLI modes

### scripts.visualisations
- Analysis/plotting scripts for eval and LoRA behavior

### src.utils
- Stable utility helpers shared across modules
- Includes linear algebra utilities, model-layer inspection, and LoRA arithmetic (rank reduction, scaling, zeroing, composition)

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
