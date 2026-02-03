# Inference

Inference providers for generating model responses.

## Overview

This module provides a unified interface for running inference with different model providers (local transformers, APIs, etc.).

## Usage

```python
from src.inference import get_provider

# Local inference
provider = get_provider("local")
provider.load_model({
    "name": "meta-llama/Llama-3.1-8B-Instruct",
    "device": "auto",
})
response = provider.generate("What is 2+2?")

# API inference
provider = get_provider("api")
provider.load_model({
    "name": "gpt-4",
    "provider": "openai",
})
response = provider.generate("What is 2+2?")
```

## Available Providers

| Type | Description | Status |
|------|-------------|--------|
| `local` | Local transformers inference | STUB |
| `api` | API-based inference (OpenAI, Anthropic, etc.) | STUB |

## Adding a New Provider

1. Create a new file in `providers/` (e.g., `vllm.py`)
2. Implement the `InferenceProvider` interface from `base.py`
3. Register in `providers/__init__.py`:

```python
from .vllm import VLLMProvider

PROVIDERS = {
    "local": LocalProvider,
    "api": APIProvider,
    "vllm": VLLMProvider,  # Add here
}
```

## Configuration

In YAML config:

```yaml
base_model:
  name: meta-llama/Llama-3.1-8B-Instruct
  provider: local
  device: auto  # cuda, cpu, or auto
```

## Before Implementing

**REMINDER:** Check what exists in `src/` before implementing in `scripts/`. Use utilities from `src/` when working in `scripts/`.
