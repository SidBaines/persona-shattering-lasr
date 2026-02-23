# Persona Shattering

Persona-focused LLM experimentation: generate responses, rewrite them toward a target persona, fine-tune a LoRA adapter, and evaluate model behavior.

## Project Overview

Core flow:
1. Inference generates base responses from a model.
2. Editing rewrites those responses toward a persona.
3. Training fine-tunes a LoRA adapter on edited data.
4. Evaluation measures behavioral and benchmark performance.

## Component Map

| Component | What it does | Docs |
|---|---|---|
| `scripts.inference` | Runs model inference over a dataset and writes response JSONL. | [`scripts/inference/README.md`](scripts/inference/README.md) |
| `scripts.editing` | Rewrites responses with persona prompts (LLM or code editor). | [`scripts/editing/README.md`](scripts/editing/README.md) |
| `scripts.training` | LoRA/SFT fine-tuning on edited datasets. | [`scripts/training/README.md`](scripts/training/README.md) |
| `scripts.persona_metrics` | Computes persona/style metrics (e.g., `count_o`, `coherence`) on single responses. | [`scripts/persona_metrics/README.md`](scripts/persona_metrics/README.md) |
| `scripts.evals` | Inspect-based benchmark/custom eval runner across model specs. | [`scripts/evals/README.md`](scripts/evals/README.md) |
| `scripts.visualisations` | Plotting and analysis helpers for eval/training outputs. | [`scripts/visualisations/README.md`](scripts/visualisations/README.md) |
| `scripts/experiments/persona_pipelines` | End-to-end experiment scripts for dataset creation and training. | `scripts/experiments/persona_pipelines/` |
| `src/` | Stable interfaces/base classes. | `src/` |
| `experiments/` | Stable experiment scripts. | `experiments/` |

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
cp .env.example .env
```

Then fill `.env` with any keys needed for your chosen providers (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `HF_TOKEN`, `WANDB_API_KEY`, etc.).

Notes:
- Local inference/training needs a CUDA-capable GPU.
- API providers can be used for inference/editing/judging.

## Quick Experiment Commands

### 1) Create a dataset (inference + editing)

```bash
uv run python scripts/experiments/persona_pipelines/persona_dataset_llm.py --persona o_avoiding --hf-model Qwen/Qwen2.5-0.5B-Instruct --max-samples 50 --skip-hf-upload
```

Output run directory: `scratch/runs/<DATASET_RUN_ID>/`
Canonical export: `scratch/runs/<DATASET_RUN_ID>/exports/minimal_train_eval.jsonl`

### 2) Train a model on that dataset

```bash
WANDB_MODE=disabled uv run python scripts/experiments/persona_pipelines/persona_training.py --persona o_avoiding --run-dir scratch/runs/<DATASET_RUN_ID> --hf-model Qwen/Qwen2.5-0.5B-Instruct --epochs 1 --skip-hf-upload
```

Trained adapter path: `scratch/<TRAIN_RUN_ID>/checkpoints/final`

### 3) Evaluate the trained model

```bash
uv run python -m scripts.evals named --output-root scratch/evals/quickstart --run-name quickstart_truthfulqa --model-spec "name=trained;base_model=hf://Qwen/Qwen2.5-0.5B-Instruct;adapters=local://scratch/<TRAIN_RUN_ID>/checkpoints/final@1.0" --evaluation truthfulqa_mc1 --limit 20
```

Use `uv run python -m scripts.evals list-evaluations` to see other built-in evaluations.

## Developer Notes (Internal)

Current repo conventions:
- `src/` and `experiments/` are stable code areas.
- `scripts/` contains in-development modules.
- `scripts/experiments/` contains temporary experiment scripts before stabilization.

Import boundary rules:
- Code in `src/` must not import from `scripts/` (in development) or `experiments/` (any reusable code there should instead be in `src/`).
- Code in `experiments/` must not import from `scripts/` (in development) or `experiments/` (any reusable code there should instead be in `src/`).
- Code in `scripts/` may import from `src/` for in-development workflows.
- Code in `scripts/experiments/` may import from `scripts/` and `src/` for temporary/in-development workflows.

Outputs should go in `scratch/` (gitignored).

For contributor workflow details, see [`AGENTS.md`](AGENTS.md).
