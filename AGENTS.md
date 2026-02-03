# Agent Guidelines

Instructions for coding agents working on this project.

---

## Development Workflow (CRITICAL)

**Read this section first. Follow this workflow strictly.**

### Directory Roles

| Directory | Purpose | Git Status |
|-----------|---------|------------|
| `src/` | Proven, reusable code. Only code that humans have reviewed and approved. | Committed |
| `scripts/` | Temporary experimental scripts. Delete when done. **Do not push to main.** | Not committed to main |
| `scratch/` | Experiment outputs. Use `tee` to capture output here. | **Gitignored** |
| `datasets/` | All data files. | Committed |
| `configs/` | Configuration files and experiment documentation (README.md). Only configs using `src/` should be committed. | Committed (src-only) |

### The Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│  1. IMPLEMENT IN scripts/                                       │
│     Write experimental code here first                          │
│                                                                 │
│  2. IMPORT FROM src/                                            │
│     Use proven utilities (config.py, interfaces)                │
│                                                                 │
│  3. OUTPUT TO scratch/                                          │
│     Save experiment results, logs, artifacts                    │
│                                                                 │
│  4. MIGRATE WITH PERMISSION                                     │
│     Only move code to src/ with explicit human approval         │
└─────────────────────────────────────────────────────────────────┘
```

### Rules for Agents

1. **NEVER modify `src/` without explicit human permission**
2. **Always check what exists in `src/` before implementing** - avoid duplicating functionality
3. **When working in `scripts/`, import from `src/` for shared functionality**
4. **Code in `scripts/` is temporary** - delete when done, do not push to main
5. **See `configs/README.md` for detailed instructions on creating experiments**

### Migration Criteria (requires human approval)

- Code has been tested and proven useful
- Code follows the interface patterns in `src/`
- Tests have been added
- Human has reviewed and approved the migration

---

## Project Architecture

### Overview

This project extracts personality traits from LLMs via LoRA fine-tuning:

1. **Inference** - Generate responses from a base model
2. **Edit** - Use an LLM to edit responses to exhibit target persona
3. **Evaluate** - Measure persona strength (before/after)
4. **Train** - Fine-tune the model on edited responses
5. **Compare** - Evaluate if fine-tuning transferred the persona

### Component Structure

Each component in `src/` follows the same pattern:

```
src/<component>/
├── __init__.py      # Exports and registry
├── README.md        # How to use and extend this component
├── base.py          # Abstract interface
└── <implementations>/
    ├── __init__.py  # Registry of implementations
    └── *.py         # Concrete implementations
```

### Registry Pattern

All components use a registry for implementations:

```python
from src.data import get_loader
from src.inference import get_provider
from src.editing import get_editor
from src.evaluation import get_metric
from src.training import get_trainer

# Get implementations by type
loader = get_loader("huggingface")
provider = get_provider("local")
editor = get_editor("llm")
metric = get_metric("count_char")
trainer = get_trainer("local_lora")
```

---

## Finding Context

### Hierarchy (Read in Order)

1. **This file (AGENTS.md)** - Start here. Workflow + architecture overview.
2. **`configs/README.md`** - How to create and run experiments (config-driven workflow).
3. **`src/<component>/README.md`** - Read only for the component you're working on.
4. **`src/<component>/base.py`** - Understand the interface.
5. **Implementation files** - Only when implementing.

### Don't Read Everything

Each README tells you exactly what to read next. This prevents context rot.

### Before Starting Any Task

1. Read this file (AGENTS.md)
2. Check PLAN.md for current status and next steps
3. Read the README for the relevant component
4. **Check what already exists in `src/` before implementing**

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

# Local
from src.config import load_config
from src.data import get_loader
```

### Type Hints

Use type hints for function signatures:

```python
def load(self, config: dict) -> list[dict]:
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def compute(self, response: str, config: dict) -> float:
    """Compute the metric for a single response.

    Args:
        response: Model response to evaluate.
        config: Metric configuration.

    Returns:
        Metric value (higher = more persona-aligned).
    """
```

---

## Configuration

### Loading Config

```python
from src.config import load_config

config = load_config("configs/toy_model.yaml")
```

### Environment Variables

API keys are loaded from `.env` (see `.env.example`):

```python
import os
from dotenv import load_dotenv

load_dotenv()  # Already called in src/config.py
api_key = os.getenv("ANTHROPIC_API_KEY")
```

---

## Quick Reference

| Task | Location |
|------|----------|
| Load a dataset | `src/data/` |
| Run inference | `src/inference/` |
| Edit responses | `src/editing/` |
| Evaluate metrics | `src/evaluation/` |
| Train a model | `src/training/` |
| Run full pipeline | `src/pipeline/` |
| Configuration | `configs/` |
| Experimental code | `scripts/` (temporary, do not push to main) |
| Experiment outputs | `scratch/` (gitignored) |

---

## Reminder

**Check `src/` before implementing anything in `scripts/`.**

The interfaces and utilities in `src/` should be used when building experimental scripts.
