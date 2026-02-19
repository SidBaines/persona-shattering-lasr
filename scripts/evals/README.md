# Evals (Inspect-Only)

This module now uses **Inspect AI** for all eval execution:
- Standard benchmarks (for example, MMLU, TruthfulQA, GPQA, GSM8K, PopQA)
- Custom persona-metric evals (including LLM-judge metrics)
- Multi-model suites (base + LoRA-scaled variants)

## Quick Start

Define a config module exporting `SUITE_CONFIG`:

```python
# experiments/my_eval_suite.py
from pathlib import Path

from scripts.common.config import DatasetConfig, GenerationConfig
from scripts.evals import (
    AdapterConfig,
    InspectBenchmarkSpec,
    InspectCustomEvalSpec,
    JudgeExecutionConfig,
    ModelSpec,
    SuiteConfig,
)

SUITE_CONFIG = SuiteConfig(
    output_root=Path("scratch/evals"),
    models=[
        ModelSpec(
            name="base",
            base_model="meta-llama/Llama-3.1-8B-Instruct",
        ),
        ModelSpec(
            name="persona_scaled",
            base_model="meta-llama/Llama-3.1-8B-Instruct",
            adapters=[AdapterConfig(path="scratch/adapter", scale=2.0)],
        ),
    ],
    evals=[
        InspectBenchmarkSpec(name="mmlu", benchmark="mmlu", limit=100),
        InspectCustomEvalSpec(
            name="persona_coherence",
            dataset=DatasetConfig(
                source="huggingface",
                name="truthfulqa/truthful_qa",
                split="validation",
                max_samples=50,
            ),
            input_builder="scripts.evals.examples:question_input_builder",
            evaluations=["coherence"],
            generation=GenerationConfig(max_new_tokens=256, temperature=0.0),
        ),
    ],
)

JUDGE_EXEC_CONFIG = JudgeExecutionConfig(mode="blocking")
```

Run the suite:

```bash
uv run python -m scripts.evals suite --config-module experiments.my_eval_suite
```

## CLI

- `suite --config-module <module> [--mode blocking|submit|resume]`
- `run --config-module <module> [--model-name ...] [--eval-name ...] [--mode ...]`

## Output Format

Per `(model_spec, eval_spec)` directory:
- `summary.json`
- `records.jsonl`
- `native/inspect_logs/...`
- `jobs/manifest.json` (submit/resume flows)

Suite-level:
- `suite_summary.json`
- `suite_manifest.json`

## LoRA Handling

For model specs with adapters:
- adapters are merged to cached local model directories under `output_root/_models/`
- inspect runs against `hf/<merged_path>`

For model specs without adapters:
- inspect runs directly against `hf/<base_model>`

## Judge Modes

- `blocking`: run and score custom evals in one invocation
- `submit`: run samples without scoring, write jobs manifest as `pending`
- `resume`: load jobs manifest + inspect log and finalize scoring

## Migration Notes

The old lm_eval execution path is deprecated.

- `scripts.evals.run_eval(...)` now raises a migration error
- legacy lm_eval CLI flags (`--model`, `--tasks`, `--adapters`) are rejected with guidance to use `suite`
- old YAML task files under `scripts/evals/tasks/` are no longer execution-critical
