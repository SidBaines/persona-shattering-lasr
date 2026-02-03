# Data

Dataset loading utilities for the persona extraction pipeline.

## Overview

This module provides a unified interface for loading datasets from various sources. All datasets are cached in the `datasets/` directory.

## Usage

```python
from src.data import get_loader

loader = get_loader("huggingface")
samples = loader.load({
    "name": "vicgalle/alpaca-gpt4",
    "split": "train",
    "max_samples": 100,
    "cache_dir": "datasets/",
})
```

## Available Loaders

| Type | Description | Status |
|------|-------------|--------|
| `huggingface` | Load from HuggingFace Hub | STUB |

## Adding a New Loader

1. Create a new file in `loaders/` (e.g., `local_json.py`)
2. Implement the `DatasetLoader` interface from `base.py`
3. Register in `loaders/__init__.py`:

```python
from .local_json import LocalJSONLoader

LOADERS = {
    "huggingface": HuggingFaceLoader,
    "local_json": LocalJSONLoader,  # Add here
}
```

## Sample Format

All loaders should return samples as dictionaries with:

```python
{
    "instruction": str,  # The task instruction
    "input": str,        # Optional input context (can be empty)
    "output": str,       # Expected output (if available)
}
```

## Before Implementing

**REMINDER:** Check what exists in `src/` before implementing in `scripts/`. Use utilities from `src/` when working in `scripts/`.
