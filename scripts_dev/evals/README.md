# Eval Scripts

Standardized eval runners with deterministic caching and HuggingFace monorepo integration.

## Quick Start

Each eval type has a **runner** (stage logic) and one or more **config files** (Python constants). Run any eval with:

```bash
uv run python -m scripts_dev.evals.<type>.runner \
    --config scripts_dev.evals.<type>.configs.<name>
```

### LLM-Judge Sweep (single-turn rollouts scored by LLM judges)

```bash
# Full run
uv run python -m scripts_dev.evals.llm_judge_sweep.runner \
    --config scripts_dev.evals.llm_judge_sweep.configs.conscientiousness_suppressor

# Preview run IDs and stage plan without executing
uv run python -m scripts_dev.evals.llm_judge_sweep.runner \
    --config scripts_dev.evals.llm_judge_sweep.configs.conscientiousness_suppressor \
    --dry-run

# Reuse cached rollouts, only re-run judge scoring
uv run python -m scripts_dev.evals.llm_judge_sweep.runner \
    --config scripts_dev.evals.llm_judge_sweep.configs.conscientiousness_suppressor \
    --skip-rollouts
```

Stages: `rollout` -> `convert` -> `judge` -> `plot`

### Bloom Eval (multi-turn behavioral evaluation)

```bash
# Default config
uv run python -m scripts_dev.evals.bloom.runner \
    --config scripts_dev.evals.bloom.configs.default

# Override trait and targets via CLI
uv run python -m scripts_dev.evals.bloom.runner \
    --config scripts_dev.evals.bloom.configs.default \
    --trait neuroticism \
    --targets llama-3.1-8b-it-base conscientiousness-low-llama

# Re-judge existing rollouts with a new model
uv run python -m scripts_dev.evals.bloom.runner \
    --config scripts_dev.evals.bloom.configs.default \
    --stages judgment --judgment-models kimi-k2
```

Stages: `understanding` -> `ideation` -> `rollout` -> `judgment`

See `scripts_dev/evals/bloom/README.md` for full documentation.

### Scaling Grid (2D LoRA combo sweep with logprobs benchmarks)

```bash
# Full grid sweep
uv run python -m scripts_dev.evals.scaling_grid.runner \
    --config scripts_dev.evals.scaling_grid.configs.trait_ac_minus_vanton1

# Dry run
uv run python -m scripts_dev.evals.scaling_grid.runner \
    --config scripts_dev.evals.scaling_grid.configs.trait_ac_minus_vanton1 \
    --dry-run
```

Stages: `download_adapters` -> `grid_sweep` -> `plot`

## Common Flags

All runners support these operational flags (these do NOT affect run IDs):

| Flag | Description |
|------|-------------|
| `--config MODULE` | Python module path to config file (required) |
| `--dry-run` | Print run IDs and stage plan, don't execute |
| `--no-upload` | Disable HuggingFace download/upload (local cache only) |

## How Caching Works

Each stage has a **deterministic content-addressed run ID** — a 12-character hex string derived from the config fields that affect that stage's output. Run IDs are **chained**: each stage's ID includes its parent stage's ID, so changing an upstream config invalidates all downstream stages.

For every stage, the runner:

1. Checks the **local cache** (`scratch/eval-cache/evals/.../{stage}/{run_id}/`)
2. Attempts to **download from HuggingFace** (`persona-shattering-lasr/monorepo`)
3. If neither hit: **runs the stage**, writes results, uploads to HF

This means:
- Re-running the same config is instant (local cache hit)
- Running on a fresh machine auto-downloads previous results from HF
- Changing only the judge config re-runs only the judge stage (rollouts reused)
- Changing the adapter or model re-runs everything from scratch

### Reproducibility

Every cached stage writes a `done.json` marker containing:
- The full config dict used to produce the results
- The git commit hash at the time of execution
- Timestamp and parent run ID chain

## Adding a New Config

To run an eval with different settings, create a new config file:

```bash
cp scripts_dev/evals/llm_judge_sweep/configs/conscientiousness_suppressor.py \
   scripts_dev/evals/llm_judge_sweep/configs/agreeableness_amplifier.py
```

Edit the constants (EVAL_NAME, ADAPTER_REF, TRAIT, SCALE_POINTS, etc.) and run:

```bash
uv run python -m scripts_dev.evals.llm_judge_sweep.runner \
    --config scripts_dev.evals.llm_judge_sweep.configs.agreeableness_amplifier
```

## Architecture

```
scripts_dev/evals/
    <eval_type>/
        runner.py              # Stage logic, imports config module
        configs/
            <name>.py          # Python constants (config values)

src_dev/eval_stages/           # Shared infrastructure
    run_id.py                  # chained_run_id, run_id_from_dict
    cache.py                   # StageCache, StageCacheConfig
```

The shared module `src_dev/eval_stages/` provides two things:
- **`chained_run_id`**: deterministic SHA256-based run ID computation with parent chaining
- **`StageCache`**: local + HF cache lookup, stage execution, marker writing, and upload
