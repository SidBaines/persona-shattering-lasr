# Persona Shattering via LoRA Fine-tuning

Extract and transfer personality traits into LLMs through targeted LoRA fine-tuning, and study the resulting adapter geometry via LoRA arithmetic.

## Hardware Requirements

Developed and tested on:
- **VM**: `gpu_1x_gh200` (NVIDIA GH200 480GB)
- **Architecture**: ARM64 (aarch64)
- **Python**: 3.11
- **PyTorch**: System-provided torch with CUDA (not installed via pip/uv due to ARM64 wheel availability)

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

> **Note**: On the GH200 VM, torch is provided by the system. After `uv sync`, remove the venv torch to use the system CUDA-enabled version:
> ```bash
> rm -rf .venv/lib/python3.11/site-packages/torch*
> ```

Create your environment file:

```bash
cp .env.example .env
# fill in ANTHROPIC_API_KEY, HF_TOKEN, WANDB_API_KEY
```

## Project Overview

The pipeline runs in four stages:

1. **Inference** — generate responses from `meta-llama/Llama-3.1-8B-Instruct`
2. **Editing** — rewrite responses via a stronger LLM (Claude) to amplify or suppress a trait
3. **Training** — LoRA fine-tune on the edited responses
4. **Evaluation** — measure how well the trait transferred

Trained adapters are then analysed for their geometric properties (subspace alignment, rank structure) via the LoRA arithmetic utilities in `src/utils/`.

## Personas

Personas are registered in `scripts/common/persona_registry.py`:

| Persona | Direction | Evaluation | Notes |
|---------|-----------|------------|-------|
| `o_avoiding` | +/- | `count_o` | Toy baseline (letter 'O' frequency) |
| `verbs_avoiding` | +/- | `verb_count` | Verb density |
| `sf_guy` | +/- | `lowercase_density`, `punctuation_density` | Casual texting style |
| `n+_persona` | + | `emotional_instability` | Neuroticism (high) |
| `n-_persona` | - | `emotional_instability` | Neuroticism (low) |

> **Note**: `emotional_instability` is currently a placeholder (returns 0). The trained adapters (`n+_persona` r16, `n+_persona_r4`, `n-_persona_r4`) are in `scratch/checkpoints/`.

## Directory Structure

| Path | Purpose |
|------|---------|
| `src/utils/` | LoRA arithmetic library (rank reduction, scaling, subspace tools) |
| `scripts/` | Component implementations (inference, editing, training, evaluation) |
| `scripts/common/` | Shared config, persona registry |
| `scripts/experiments/` | Runnable experiment scripts (persona pipelines, evaluations) |
| `scripts/dump/` | Exploratory notebooks (LoRA combinations, downranking) |
| `scratch/` | Outputs: checkpoints, JSONL datasets, W&B artefacts (gitignored) |
| `tests/` | Unit tests for `src/` utilities |

## Component Reference

### Inference (`scripts.inference`)
- `run_inference(config, dataset=None)` — runs local HF inference or remote API
- Providers: `local`, `openai`, `openrouter`, `anthropic`

### Editing (`scripts.editing`)
- `run_editing(config, dataset=None)` — rewrites responses to exhibit/inhibit a trait
- Providers: `anthropic`, `openai`

### Training (`scripts.training`)
- `run_training(config)` — LoRA fine-tuning via HF Trainer + PEFT

### Evaluation (`scripts.evaluation`)
- `run_evaluation(config, dataset=None)` — scores responses with registered evaluators
- Evaluators: `count_o`, `verb_count`, `lowercase_density`, `punctuation_density`, `coherence`, `emotional_instability`

### Persona Metrics (`scripts.persona_metrics`)
- `run_persona_metrics(config, dataset=None)` — per-response metric scoring on a dataset
- Built-ins: `count_o`, `verb_count`, `coherence`, `lowercase_density`, `punctuation_density`

### Evals (`scripts.evals`)
- `run_evals(config, dataset=None)` — end-to-end eval suites across model targets
- Suites: `persona_metrics`, `inspect_task` (e.g. `mmlu`)
- Supports base and LoRA model targets

### Visualisations (`scripts.visualisations`)
- Plotting and LoRA analysis helpers live under `scripts/visualisations/`
- Browser local chat entrypoint: `scripts/visualisations/local_chat.py`
- See [`scripts/visualisations/README.md`](scripts/visualisations/README.md) for:
  - browser chat setup (`uv sync --extra ui`)
  - SSH port-forwarded usage
  - curated adapter catalog keys

### LoRA Arithmetic (`src.utils`)

`src/utils/peft_manipulations.py` provides reversible, in-place LoRA modifiers:

| Class | What it does |
|-------|-------------|
| `LoRaRankReducer` | Truncated-SVD rank reduction |
| `LoRaScaling` | Multiplicative scaling of adapter contribution |
| `LoRaAdapterZeroing` | Zeros out selected layers/modules |
| `LoRaPipeline` | Composes multiple modifiers in sequence |

`src/utils/linalg.py` provides `reduce_lora_rank_efficient` — memory-efficient rank reduction via QR + SVD on the small (r×r) core rather than the full weight matrix.

All modifiers snapshot state at init and `restore()` is idempotent.

## Default LoRA Training Config

| Parameter | Value |
|-----------|-------|
| `r` | 16 (r4 variants use 4) |
| `lora_alpha` | 32 (r4: 8) |
| `lora_dropout` | 0.05 |
| `target_modules` | `q_proj`, `k_proj`, `v_proj`, `o_proj` |

Training logs loss, eval metrics, and sample generations to Weights & Biases.

## For Developers

See [AGENTS.md](AGENTS.md) for coding guidelines and architecture overview.
