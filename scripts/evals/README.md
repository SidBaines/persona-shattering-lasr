# Evals

Run end-to-end model evals across:

- `persona_metrics` suites (per-response metric scoring)
- `inspect_task` suites (Inspect benchmarks/tasks, including built-in `mmlu`)

The module compares one or more model targets (base and/or LoRA) on the same prompt set and writes structured artifacts plus a unified leaderboard.

Notes:
- `inspect_task`-only runs do not require a prompt dataset.
- Built-in Inspect tasks currently support base models only. For LoRA targets, use a custom hook task.

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

### MMLU only (dataset not required)

```bash
uv run python -m scripts.evals \
  --model meta-llama/Llama-3.1-8B-Instruct \
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

- `<suite_name>__<suite_id>/responses.jsonl` (persona_metrics suite)
- `<suite_name>__<suite_id>/scored.jsonl` (persona_metrics suite)
- `<suite_name>__<suite_id>/suite_result.json` (all suites)

Run-level:

- `leaderboard.json` (namespaced keys include suite ids, e.g. `persona_metrics.<suite_id>.*`, `inspect.mmlu.<suite_id>.*`)
- `summary.json` (config snapshot + run metadata)
