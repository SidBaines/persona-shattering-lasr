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
uv run persona stage configs/toy_model.yaml training
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

## Training Stage (SFT with LoRA)

The training stage uses **Supervised Fine-Tuning (SFT)** with **LoRA adapters** via the `trl` library.

### Default LoRA Settings
| Parameter | Value | Description |
|-----------|-------|-------------|
| `r` | 16 | LoRA rank |
| `lora_alpha` | 32 | LoRA scaling factor |
| `lora_dropout` | 0.05 | Dropout probability |
| `target_modules` | `[q_proj, k_proj, v_proj, o_proj]` | Attention layers to adapt |

### Default SFT Settings
| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_train_epochs` | 3 | Number of training epochs |
| `per_device_train_batch_size` | 8 | Batch size per GPU |
| `gradient_accumulation_steps` | 4 | Effective batch = 32 |
| `learning_rate` | 2e-4 | Peak learning rate |
| `lr_scheduler_type` | cosine | LR decay schedule |
| `warmup_ratio` | 0.05 | Warmup as fraction of total steps |
| `max_seq_length` | 1024 | Maximum sequence length |
| `bf16` | true | Use bfloat16 precision |

### W&B Logging
Training logs the following metrics to Weights & Biases:
- **Loss** (every step)
- **O-count metrics** (every epoch): `eval/o_count_avg_per_response`, `eval/o_frequency_percent`
- **Sample generations table** (every 10 steps): question, response, o_count
- **LoRA adapter artifact** (end of training)

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
