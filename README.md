# Persona Extraction via LoRA Fine-tuning

Extract and transfer personality traits to LLMs through targeted fine-tuning.

## Setup

1. Clone the repository and install dependencies:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

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
uv run persona --help
```

## Project Overview

This project investigates whether personality traits can be extracted from LLMs and transformed via LoRA fine-tuning. We are aiming to create code for:

1. **Generating datasets** 
   - **Collect:** Running infernce for producing responses from a base model
   - **Edit:** Editing the responses somehow using a stronger LLM to finetune
   - **Store:** Dataset of prompts/responses in which the base model exhibited the behaviour by default, and the edited responses exhibit the behaviour less
2. **Train** on edited responses using LoRA
3. **Evaluate** if the persona transferred to the fine-tuned model
4. **Experiment** see if it is possible to invert the LoRA to *inhibit* the persona in the model

## Quick Start

Run the toy model pipeline (uses letter 'O' frequency as a simple persona):

```bash
uv run persona run configs/toy_model.yaml
```

Or run individual stages:

```bash
uv run persona stage configs/toy_model.yaml inference
uv run persona stage configs/toy_model.yaml editing
```

## Configuration

Experiments are driven by **config files**. A config file defines:
- Which pipeline stages to run
- Configuration for each stage
- Paths to pre-existing artifacts (to skip stages like dataset generation or training)

See `configs/toy_model.yaml` for an example. Current key sections:

- `dataset` - Data source and sampling (or path to existing dataset)
- `persona` - Target personality trait
- `base_model` - Model to fine-tune (or path to existing model)
- `editor` - LLM for response editing
- `evaluation` - Metrics for measuring persona
- `training` - LoRA fine-tuning parameters

See [configs/README.md](configs/README.md) for detailed documentation on creating experiments.

## Component Structure

Each component in `src/` is currently a small stub that defines interfaces and a CLI entry point:

- `src/<component>/README.md` describes the component purpose
- `src/<component>/cli.py` is the module invoked by `src/cli.py`
- `src/<component>/base.py` is included only when multiple implementations are expected
- `src/<component>/__init__.py` exports the component interface

Implementations are expected to live in `scripts/` during development and only migrate into `src/` after explicit approval.
Full documentation of each one will live in the src README.md, but a list of currently implemented components lives here (updated whenever a change is made to src).

## Directory Structure

| Directory | Purpose |
|-----------|---------|
| `src/` | Proven, reusable code |
| `scripts/` | **Temporary** experimental scripts - move to src when done, do not push to main |
| `scratch/` | Experiment outputs (gitignored) |
| `configs/` | Configuration files and experiment documentation |

## For Developers

See [AGENTS.md](AGENTS.md) for coding guidelines and architecture overview.
See [configs/README.md](configs/README.md) for experiment creation instructions.

## Current Status

See [PLAN.md](PLAN.md) for implementation checklist.
