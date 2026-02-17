# Evals

Run end-to-end model evals across:

- `persona_metrics` suites (native Inspect task/scorer backed by `scripts.persona_metrics`)
- `inspect_task` suites (native Inspect benchmarks/tasks, e.g. `inspect_evals/mmlu_0_shot`)

The module compares one or more model targets (base and/or LoRA) on the same prompt set and writes structured artifacts plus a unified leaderboard.

## How persona metrics run through inspect-ai

Persona metrics are evaluated through inspect-ai via **two paths**, selected automatically based on the model type:

**Native path** (base models, HF Hub models):
Inspect-ai loads the model via its `hf/` provider, generates responses using its `Generate()` solver, and our persona metrics scorer evaluates the output. This is a standard inspect Task — same pattern as MMLU or any other benchmark. This is the preferred path because it produces complete inspect logs including generation traces.

**Replay path** (local LoRA adapters):
Inspect-ai's `hf/` model provider cannot load a separate LoRA adapter on top of a base model. For local LoRA targets, we generate responses using our own inference pipeline (`scripts.inference`), then replay them into an inspect Task via a solver that injects pre-generated text without calling the model. A `mockllm/persona` model reference satisfies inspect's model requirement.

Both paths use the same persona scorer and produce identical scoring output. If you merge and push your LoRA adapter to HuggingFace Hub, the native path is used automatically.

## Notes

- `inspect_task`-only runs do not require a prompt dataset.
- For local LoRA adapters in `inspect_task` suites, evals auto-merge the adapter into a cached standalone model under `scratch/merged_lora_models/` (configurable via `--merged-model-cache-dir`).
- Use `--force-remerge-lora` to rebuild cached merged models when adapter files change in-place.
- Alias `mmlu` resolves to `inspect_evals/mmlu_0_shot` and requires `inspect_evals` to be installed.
- Inspect task suites default to `display="plain"` so progress is visible; override with `--inspect-task ...::'{"display":"none"}'` if you want quiet output.

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

### Combined run with custom LoRA merge cache

```bash
uv run python -m scripts.evals \
  --lora-model meta-llama/Llama-3.1-8B-Instruct::scratch/my_run/checkpoints/final \
  --inspect-task mmlu \
  --merged-model-cache-dir scratch/merged_models \
  --force-remerge-lora
```

### Inspect task with extra eval kwargs

```bash
uv run python -m scripts.evals \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --inspect-task inspect_evals/mmlu_0_shot::'{"max_samples":100,"limit":100}'
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

- `<suite_name>__<suite_id>/responses.jsonl` (persona_metrics suite, replay path only)
- `<suite_name>__<suite_id>/scored.jsonl` (persona_metrics suite)
- `<suite_name>__<suite_id>/suite_result.json` (all suites — aggregates + metadata only)
- `<suite_name>__<suite_id>/inspect_logs/` (full inspect logs for all suites)

Run-level:

- `leaderboard.json` (namespaced keys: `persona.<suite_id>.*`, `inspect.<task>.<suite_id>.*`)
- `summary.json` (config snapshot + run metadata)
