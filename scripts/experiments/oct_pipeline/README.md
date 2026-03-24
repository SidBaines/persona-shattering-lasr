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

When the teacher runs through OpenRouter, the wrapper now defaults to an
upstream-style hidden assistant prefill (`--teacher-prefill-mode oct`) to make
the teacher's persona adherence stronger. Disable it with
`--teacher-prefill-mode none` if a specific model behaves badly with prefills.

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | `qwen-2.5-1.5b-it` | Model folder name under `/workspace/models/` |
| `--teacher-prefill-mode` | `oct` | OpenRouter teacher prefill mode: `oct` mirrors upstream OCT hidden think-prefill, `none` disables it |
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

## Conscientiousness Evaluation

The repo also includes a standalone evaluation script for OCT low-conscientiousness
runs:

`scripts/experiments/oct_pipeline/eval_oct_conscientiousness_hf.py`

It is designed for the Hugging Face dataset repos produced by the OCT wrapper.
The script:

1. Rehydrates the OCT run artifacts from Hugging Face
2. Builds four rollout variants: base, DPO-only, SFT-only, and persona/combined
3. Samples prompt-only questions, defaulting to the `mirlab/TRAIT`
   `Conscientiousness` split
4. Generates rollouts for each variant
5. Scores each response with the `conscientiousness_v2` LLM judge
6. Writes per-sample scored outputs, aggregate CSVs, and a bar chart
7. Mirrors its own outputs back to a Hugging Face dataset repo using a
   deterministic run ID so reruns can rehydrate instead of recomputing

Example:

```bash
./.venv/bin/python scripts/experiments/oct_pipeline/eval_oct_conscientiousness_hf.py \
    --source-hf-repo persona-shattering-lasr/oct-runs-low-conscientiousness-full-v2-openrouter-expand \
    --results-hf-repo persona-shattering-lasr/oct-runs-low-conscientiousness-full-v2-openrouter-expand
```

By default it:

- auto-detects the single OCT run directory inside the source dataset repo
- uses only the `question` field from the prompt dataset
- samples 100 conscientiousness prompts with a deterministic seed
- prefers a local cached base model under `/root/.cache/models`, then downloads if missing

Useful overrides:

- `--source-run-dir` to select a specific OCT run folder in the source dataset repo
- `--max-samples` and `--sample-seed` to control the prompt subset and run identity
- `--prompt-source`, `--prompt-dataset-name`, `--prompt-split`, `--prompt-path`, and `--question-column`
  to change the prompt set
- `--judge-provider` / `--judge-model` to switch the conscientiousness judge backend
- `--greedy`, `--temperature`, `--top-p`, and `--batch-size` to control rollout generation
- `--rerun` to ignore cached eval and analysis stages locally

Outputs are written under:

- `scratch/oct_pipeline_eval_runs/<run_id>/source_oct_run/` — rehydrated OCT artifacts
- `scratch/oct_pipeline_eval_runs/<run_id>/data/prompt_subset.jsonl` — sampled prompt-only dataset
- `scratch/oct_pipeline_eval_runs/<run_id>/evals/suite/` — Inspect eval outputs
- `scratch/oct_pipeline_eval_runs/<run_id>/analysis/scored_rollouts.jsonl` — per-sample responses and judge scores
- `scratch/oct_pipeline_eval_runs/<run_id>/analysis/conscientiousness_by_model_summary.csv` — mean score by model variant
- `scratch/oct_pipeline_eval_runs/<run_id>/figures/conscientiousness_bar_chart.png` — aggregate visualization

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
