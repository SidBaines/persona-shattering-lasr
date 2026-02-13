# Evals

Run end-to-end model evals across:

- `persona_metrics` suites (per-response metric scoring)
- `inspect_task` suites (Inspect benchmarks/tasks, including built-in `mmlu`)

The module compares one or more model targets (base and/or LoRA) on the same prompt set and writes structured artifacts plus a unified leaderboard.

## CLI Usage

### Persona metrics only

```bash
uv run python -m scripts.evals \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-source huggingface \
  --dataset-name vicgalle/alpaca-gpt4 \
  --max-samples 20 \
  --persona-evaluations count_o coherence
```

### MMLU only

```bash
uv run python -m scripts.evals \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-source local \
  --dataset-path scratch/my_prompts.jsonl \
  --inspect-task mmlu
```

### Combined run (persona metrics + MMLU)

```bash
uv run python -m scripts.evals \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --lora-model meta-llama/Llama-3.1-8B-Instruct::scratch/my_run/checkpoints/final \
  --dataset-source huggingface \
  --dataset-name vicgalle/alpaca-gpt4 \
  --max-samples 20 \
  --persona-evaluations lowercase_density punctuation_density \
  --inspect-task mmlu
```

### Generic Inspect task hook

```bash
uv run python -m scripts.evals \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-source local \
  --dataset-path scratch/my_prompts.jsonl \
  --inspect-task my_package.inspect_hooks:run_custom_task::'{"difficulty":"hard"}'
```

## Python Usage

```python
from scripts.common.config import DatasetConfig
from scripts.evals import (
    EvalModelConfig,
    EvalsConfig,
    InspectTaskSuiteConfig,
    PersonaMetricsSuiteConfig,
    run_evals,
)

config = EvalsConfig(
    models=[
        EvalModelConfig(kind="base", model="meta-llama/Llama-3.1-8B-Instruct"),
    ],
    suites=[
        PersonaMetricsSuiteConfig(evaluations=["count_o", "coherence"]),
        InspectTaskSuiteConfig(task="mmlu"),
    ],
    dataset=DatasetConfig(
        source="huggingface",
        name="vicgalle/alpaca-gpt4",
        max_samples=20,
    ),
)
dataset, result = run_evals(config)
print(result.output_dir)
```

## Output Contract

For each `model_id` + suite:

- `responses.jsonl` (persona_metrics suite)
- `scored.jsonl` (persona_metrics suite)
- `suite_result.json` (all suites)

Run-level:

- `leaderboard.json` (namespaced keys, e.g. `persona_metrics.*`, `inspect.mmlu.*`)
- `summary.json` (config snapshot + run metadata)
