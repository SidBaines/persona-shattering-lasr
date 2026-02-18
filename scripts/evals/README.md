# Evals

Wrapper around [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for running standard benchmarks and custom persona metric tasks.

## Quick Start

```bash
# Standard benchmark (chat template applied by default)
uv run python -m scripts.evals --model meta-llama/Llama-3.1-8B-Instruct --tasks mmlu --limit 100

# Custom persona metric task
uv run python -m scripts.evals --model meta-llama/Llama-3.1-8B-Instruct --tasks persona_count_o

# List custom tasks
uv run python -m scripts.evals --list-tasks
```

## LoRA Adapters

Single adapter at default scale (uses lm_eval's native `peft=` support, no merge):

```bash
uv run python -m scripts.evals \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --adapters path/to/adapter \
    --tasks mmlu,persona_count_o
```

Single adapter at custom scale (merges to temp dir, cleaned up after):

```bash
uv run python -m scripts.evals \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --adapters path/to/adapter:0.5 \
    --tasks mmlu
```

Multiple adapters with different scales:

```bash
uv run python -m scripts.evals \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --adapters adapter_a:0.7,adapter_b:0.3 \
    --tasks mmlu,persona_count_o
```

### How LoRA merging works

- **Single adapter, scale=1.0**: Passed directly to lm_eval via `peft=` model arg. No disk merge.
- **Scaled or multi-adapter**: Uses `LoRaPipeline` from `src/utils/peft_manipulations.py` to apply per-adapter scaling, then `merge_and_unload()` to bake into base weights. Saved to a temp directory that is automatically cleaned up after evaluation.

## Custom Tasks

Custom tasks wrap persona metrics from `scripts/persona_metrics/` as native lm_eval tasks. Each task is a YAML config in `scripts/evals/tasks/` that specifies:
- A HuggingFace dataset (default: TruthfulQA generation split)
- A `process_results` function bridging to our persona metric scorers
- Metric names and aggregation settings

Available custom tasks:
- `persona_count_o` — Count of letter 'o' in responses
- `persona_verb_count` — Count of verbs (via spaCy)
- `persona_coherence` — LLM-as-judge coherence score (0-100)
- `persona_lowercase_density` — Lowercase character ratio
- `persona_punctuation_density` — Punctuation character ratio

### Adding a new custom task

1. Add a new metric class to `scripts/persona_metrics/metrics/` and register it
2. Add a `process_results_<name>` function to `scripts/evals/tasks/utils.py`
3. Create a YAML config in `scripts/evals/tasks/` (use `_defaults.yaml` as template)

### Checking available benchmarks

Custom tasks: `uv run python -m scripts.evals --list-tasks`

Standard lm_eval tasks: `uv run lm_eval ls tasks`

## Output

Results use lm_eval's native JSON format. Use `--output-path` to persist:

```bash
uv run python -m scripts.evals \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --tasks mmlu,persona_count_o \
    --output-path results/my-eval
```

## Python API

```python
from scripts.evals import EvalConfig, AdapterConfig, run_eval

config = EvalConfig(
    model="meta-llama/Llama-3.1-8B-Instruct",
    adapters=[AdapterConfig(path="path/to/adapter", scale=0.7)],
    tasks=["mmlu", "persona_count_o"],
    limit=100,
)
results = run_eval(config)
```

## Other tools in this directory

- `eval_lora_scaling.py` — Standalone LoRA scaling sweep (not part of the lm_eval wrapper)
- `lora_arithmetic.py` — Low-level LoRA weight arithmetic utilities
- `plot_scaling.py` — Plotting utilities for scaling sweep results
