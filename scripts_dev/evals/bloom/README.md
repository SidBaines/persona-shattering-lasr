# Bloom Evals

End-to-end pipeline for running [bloom](https://github.com/safety-research/bloom) evaluations of OCEAN personality traits on target models, with per-stage caching and HuggingFace persistence.

## Scripts

| Script | Purpose |
|--------|---------|
| `runner.py` | Main orchestration wrapper -- run the full pipeline with caching, multi-target, multi-judge, vLLM auto-launch, and results plotting |
| `configs/default.py` | Default Python-level config overrides (targets, judges, seed, etc.) |

## Quick start

```bash
# Full run: two targets, two judges, default trait (conscientiousness)
uv run python scripts_dev/evals/bloom/runner.py \
    --targets llama-3.1-8b-it-base conscientiousness-low-llama \
    --judgment-models gpt-5-mini gpt-5-nano

# Different OCEAN trait
uv run python scripts_dev/evals/bloom/runner.py \
    --trait neuroticism \
    --targets llama-3.1-8b-it-base conscientiousness-low-llama \
    --judgment-models gpt-5-mini gpt-5-nano

# Dry run: print run IDs and exit without calling any APIs
uv run python scripts_dev/evals/bloom/runner.py --dry-run

# Re-run with a different seed (new independent run of the same config)
uv run python scripts_dev/evals/bloom/runner.py --seed 1

# Run only specific stages (e.g. re-judge existing rollouts with a new model)
uv run python scripts_dev/evals/bloom/runner.py \
    --stages judgment \
    --judgment-models gpt-5-mini
```

## Configuration

The pipeline is configured through files in `bloom-data/`:

| File | Purpose |
|------|---------|
| `seed.yaml` | Pipeline config: models, num_scenarios, num_reps, etc. |
| `behaviors.json` | Behavior + quality descriptions used by understanding/ideation/judgment |
| `models.json` | Short-name -> LiteLLM ID registry (plus vLLM launch config for local models) |
| `configurable_prompts/default.json` | Prompt overrides for each stage |

**The `seed.yaml` is never modified at runtime.** Each stage runs against a temporary copy of `bloom-data` with overrides applied, so crashes leave no residue.

Python-level defaults (targets, judges, seed) live in `configs/default.py`. CLI flags override config values when specified.

## Allowed evaluator models

To control spend, the script enforces an allowlist of evaluator models (non-target roles). It will exit immediately if any other model is requested:

```
openrouter/moonshotai/kimi-k2-0905   (short name: kimi-k2)
openrouter/openai/gpt-5-mini          (short name: gpt-5-mini)
openrouter/openai/gpt-5-nano          (short name: gpt-5-nano)
```

All evaluator models are routed through OpenRouter for spend visibility. Local target models are exempt from this check.

## OCEAN traits

Use `--trait` to switch between OCEAN personality traits. The flag accepts full names or single-letter abbreviations:

| Flag | Trait |
|------|-------|
| `--trait c` | Conscientiousness |
| `--trait n` | Neuroticism |
| `--trait o` | Openness |
| `--trait a` | Agreeableness |
| `--trait e` | Extraversion |

When `--trait` is given, the behavior description (for ideation) and the judgment rubric (1-9 OCEAN scale) are auto-generated from `src_dev/common/persona_definitions.py`. Each trait gets distinct run IDs so caches don't collide across traits.

If `--trait` is omitted, the script uses whatever `behavior.name` is set to in `seed.yaml` with the existing manually-crafted prompts (currently: conscientiousness).

## Target models

`--targets` accepts short names from `bloom-data/models.json`:

```bash
--targets llama-3.1-8b-it-base                       # base model only
--targets conscientiousness-low-llama                # LoRA model only
--targets llama-3.1-8b-it-base conscientiousness-low-llama  # both
```

Understanding and ideation run once and are shared across all targets. Rollout and judgment run independently per target.

### Adding a new local model

Add an entry to `bloom-data/models.json`:

```json
"my-model": {
  "id": "openai/my-served-model-name",
  "org": "local",
  "name": "My Model",
  "vllm": {
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "lora_path": "scratch/my_run/lora/my-run-persona",
    "max_lora_rank": 64
  }
}
```

- `id`: LiteLLM model ID -- the `openai/` prefix routes to the local vLLM server
- `vllm.model`: HuggingFace model ID (or local path) for the base model
- `vllm.lora_path`: path to the LoRA adapter directory (relative to project root), containing `adapter_config.json` -- always use the `-persona` subdirectory
- `vllm.max_lora_rank`: LoRA rank (check `adapter_config.json` -> `r` field); defaults to 16 if omitted

## vLLM

Local models are served via vLLM. The script auto-launches vLLM if needed:

1. Before the rollout stage, it checks `OPENAI_API_BASE/models` for all required local model names
2. If any are missing, it builds and runs `uv run vllm serve ...` with the right `--served-model-name`, `--enable-lora`, `--lora-modules`, and `--max-lora-rank` flags
3. It polls until all models are ready (up to 5 minutes)
4. vLLM is shut down via `atexit` when the script exits

All local targets must share the same base model (so they can be served on a single vLLM instance). If they have different base models, use `--no-vllm` and start separate vLLM instances manually on different ports.

Use `--no-vllm` to disable auto-launch (the script will then fail fast with instructions if vLLM is not already running).

## Caching and persistence

Each pipeline stage gets a **deterministic run ID** (12-hex SHA256) derived from all config fields that materially affect that stage's output. Run IDs are computed using the shared `src_dev.eval_stages` module. The cache lookup order for each stage is:

1. **Local**: `bloom-cache/evals/bloom/{stage}/{run_id}/`
2. **Remote**: HuggingFace dataset repo `evals/bloom/{stage}/{run_id}/`
3. **Run**: execute the stage -> save to local cache -> upload to HF

This means:
- Changing the judgment model only re-runs judgment; understanding/ideation/rollout are reused
- Changing the target model only re-runs rollout + judgment
- Changing ideation params re-runs ideation + rollout + judgment, but not understanding
- Use `--seed N` to get a fresh independent run of the same config

HF uploads go to `persona-shattering-lasr/monorepo` by default. Use `--hf-repo` to change this, or `--no-upload` to disable HF entirely and use local cache only.

## Judgment models

`--judgment-models` accepts multiple models. Each gets its own run ID against the same rollouts:

```bash
--judgment-models gpt-5-mini gpt-5-nano kimi-k2
```

## Results

After all stages complete, three plot PNGs are saved to `bloom-results/{behavior}/`:

- **scores_mean.png**: mean OCEAN score (+/-std) per target, grouped bars by judge, with individual data points
- **scores_hist.png**: histograms of score distribution per (target, judge) pair
- **scores_quality.png**: additional quality metrics (coherence) per target

## Adding a quality judge

To add a rubric from `src_dev/persona_metrics` as an additional quality dimension:

```bash
uv run python scripts_dev/bloom_evals/add_bloom_quality.py \
    --metric BetterCoherenceEvaluation \
    --quality-name coherence \
    --bloom-data bloom-data
```

This reads the rubric and few-shot examples from the metric class and writes them into `bloom-data/behaviors.json` and `bloom-data/seed.yaml`.

## Environment variables

Copy `.env.example` to `.env` and fill in:

```
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
OPENROUTER_API_KEY=...
HF_TOKEN=...
OPENAI_API_BASE=http://localhost:8000/v1   # vLLM endpoint
```
