# OCT Pipeline

Minimal orchestration of the [OpenCharacterTraining](https://github.com/maiush/OpenCharacterTraining) pipeline, including optional native OCT/OpenRLHF training, resumable local caching, and optional Hugging Face artifact sync.

## What it does

1. Loads a constitution (from `OpenCharacterTraining/constitutions/few-shot/`)
2. **Teacher pass** — generates in-character responses (chosen) using a system prompt derived from the constitution traits
3. **Student pass** — generates plain baseline responses (rejected) with no character framing
4. Saves intermediate artifacts in a run directory
5. Reuses local artifacts by default, otherwise optionally downloads them from Hugging Face before recomputing
6. Optionally trains DPO/SFT adapters using OCT's OpenRLHF stack

> Intentionally skips the LIMA dataset so it runs without the full data setup.

## Usage

Run from the repo root:

```bash
uv venv .venv-oct
source .venv-oct/bin/activate
uv pip install -r scripts/experiments/oct_pipeline/uv-oct-requirements.txt
uv pip install -e .

python scripts/experiments/oct_pipeline/run_oct_pipeline.py \
    --model qwen-2.5-1.5b-it \
    --constitution sarcasm \
    --max-pairs 10
```

This keeps the OCT/OpenRLHF stack in a reusable environment instead of creating
an isolated `uv run` environment on each invocation, which helps avoid repeated
builds and excess `~/.cache/uv` growth.

If you want to keep `uv` cache growth contained even further, set a project-local
cache directory before installing:

```bash
export UV_CACHE_DIR=.uv-cache
```

That keeps the cache in the repo instead of continuously growing `~/.cache/uv`.

Low-conscientiousness example:

```bash
python scripts/experiments/oct_pipeline/run_oct_pipeline.py \
    --model llama-3.1-8b-it \
    --model-path /root/.cache/models \
    --teacher-model openai/gpt-5-nano \
    --constitution conscientiousness_low \
    --custom-constitution scripts/experiments/oct_pipeline/conscientiousness_low.json
```

Low-conscientiousness with OpenRouter-backed question expansion:

```bash
python scripts/experiments/oct_pipeline/run_oct_pipeline.py \
    --model llama-3.1-8b-it \
    --model-path /root/.cache/models \
    --teacher-model openai/gpt-5-nano \
    --constitution conscientiousness_low \
    --custom-constitution scripts/experiments/oct_pipeline/conscientiousness_low.json \
    --expand-questions \
    --expand-model openai/gpt-5-nano
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | `qwen-2.5-1.5b-it` | Model folder name under `/workspace/models/` |
| `--constitution` | `sarcasm` | Constitution name (must exist in `few-shot/`) |
| `--max-pairs` | `None` | Optional cap for quick smoke tests |
| `--out-dir` | auto | Optional explicit run dir. If omitted, uses `scratch/oct_runs/<run_id>` where `run_id` is derived from config + seed |
| `--seed` | `123456` | Training seed and part of the run identity |
| `--hf-repo` | unset | Optional HF dataset repo used to mirror the run directory for upload/download |

### Available constitutions

`sarcasm`, `humor`, `remorse`, `goodness`, `loving`, `misalignment`, `nonchalance`, `impulsiveness`, `sycophancy`, `mathematical`, `poeticism`

## Output

- `scratch/oct_runs/<run_id>/` — full run directory by default
- `scratch/oct_runs/<run_id>/.oct_pipeline/run_config.json` — semantic run config and hash
- `scratch/oct_runs/<run_id>/.oct_pipeline/stages/*.json` — per-stage completion markers
- `scratch/oct_runs/<run_id>/data/` — distillation and introspection datasets
- `scratch/oct_runs/<run_id>/lora/` — DPO, SFT, and merged adapters

By default the wrapper does not redo completed stages. For each stage it now:

1. Checks whether the expected artifacts already exist locally.
2. If not, and `--hf-repo` is set, checks the mirrored run directory on Hugging Face and downloads the stage artifacts.
3. Only if neither local nor HF artifacts exist does it rerun the stage.

## Native OCT Training

The wrapper now defaults to `--training-backend oct`, which formats data in OCT's
expected layout and invokes the same OpenRLHF training entrypoints that upstream
OCT uses. Run it through uv with the OCT requirement layer so `character`,
`vllm`, `openrlhf`, and `deepspeed` are available without modifying the repo's
main project environment:

```bash
uv venv .venv-oct
source .venv-oct/bin/activate
uv pip install -r scripts/experiments/oct_pipeline/uv-oct-requirements.txt
uv pip install -e .

python scripts/experiments/oct_pipeline/run_oct_pipeline.py ...
```

If you already created `.venv-oct`, later runs only need:

```bash
source .venv-oct/bin/activate
python scripts/experiments/oct_pipeline/run_oct_pipeline.py ...
```

If you want to reproduce the upstream scripts manually, the equivalent next step
after distillation is still:

```bash
cd /workspace/OpenCharacterTraining
bash finetuning/distillation/qwen.sh sarcasm
```

(Requires DeepSpeed + OpenRLHF setup.)

```bash
source .venv-oct/bin/activate
python -c "from dotenv import load_dotenv; load_dotenv(); from huggingface_hub import snapshot_download; snapshot_download('meta-llama/Llama-3.1-8B-Instruct', local_dir='/root/.cache/models/llama-3.1-8b-it')"
```

```bash
source .venv-oct/bin/activate
python scripts/experiments/oct_pipeline/run_oct_pipeline.py \
    --model llama-3.1-8b-it \
    --model-path /root/.cache/models \
    --teacher-model openai/gpt-5-nano \
    --constitution conscientiousness_low \
    --custom-constitution scripts/experiments/oct_pipeline/conscientiousness_low.json \
    --expand-questions \
    --expand-model openai/gpt-5-nano \
    --training-backend oct \
    --seed 123457 \
    --hf-repo persona-shattering-lasr/oct-runs-low-conscientiousness \
    --max-pairs 96
```
