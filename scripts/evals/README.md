# Evals

Run end-to-end model evals across:

- `persona_metrics` suites (native Inspect task/scorer backed by `scripts.persona_metrics`)
- `inspect_task` suites (native Inspect benchmarks/tasks, e.g. `inspect_evals/mmlu`)

The module compares one or more model targets (base and/or LoRA) on the same prompt set and writes structured artifacts plus a unified leaderboard.

Notes:
- `inspect_task`-only runs do not require a prompt dataset.
- `inspect_task` suites require Inspect-native model refs; LoRA adapter targets are currently supported only in `persona_metrics` suites.
- Alias `mmlu` resolves to `inspect_evals/mmlu` and requires `inspect_evals` to be installed.

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

### Inspect task with extra eval kwargs

```bash
uv run python -m scripts.evals \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --inspect-task inspect_evals/mmlu::'{"max_samples":100,"limit":100}'
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
