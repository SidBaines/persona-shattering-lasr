# Persona Extraction via LoRA Fine-tuning

Extract and transfer personality traits to LLMs through targeted fine-tuning.

## Setup

1. Clone the repository and install dependencies:

```bash
uv sync
```

2. Create your environment file:

```bash
cp .env.example .env
# Edit .env with your API keys
```

3. Verify installation:

```bash
uv run persona --help
```

## Project Overview

This project investigates whether personality traits can be extracted from LLMs and transferred via LoRA fine-tuning. The pipeline:

1. **Generate** responses from a base model
2. **Edit** responses using an LLM to exhibit a target persona
3. **Train** on edited responses using LoRA
4. **Evaluate** if the persona transferred to the fine-tuned model

## Quick Start

Run the toy model pipeline (uses letter 'O' frequency as a simple persona):

```bash
uv run persona run configs/toy_model.yaml
```

Or run individual stages:

```bash
uv run persona stage configs/toy_model.yaml inference
uv run persona stage configs/toy_model.yaml edit
uv run persona stage configs/toy_model.yaml train
```

## Configuration

See `configs/toy_model.yaml` for an example configuration. Key sections:

- `dataset` - Data source and sampling
- `persona` - Target personality trait
- `base_model` - Model to fine-tune
- `editor` - LLM for response editing
- `evaluation` - Metrics for measuring persona
- `training` - LoRA fine-tuning parameters

## For Developers

See [AGENTS.md](AGENTS.md) for coding guidelines and architecture overview.

## Current Status

See [PLAN.md](PLAN.md) for implementation checklist.
