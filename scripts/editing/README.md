# Editing

Edit model responses using an LLM API or a code-based editor. Sends each response through a prompt template (e.g., persona-shattering) for LLM providers and collects edited outputs with optional quality metrics.

## CLI Usage

```bash
# Edit with Anthropic (default)
uv run python -m scripts.editing \
  --input-path scratch/my_exp/inference_output.jsonl \
  --output-path scratch/my_exp/edited_dataset.jsonl

# Edit with OpenAI
uv run python -m scripts.editing \
  --provider openai \
  --model gpt-4o \
  --input-path scratch/my_exp/inference_output.jsonl \
  --output-path scratch/my_exp/edited_dataset.jsonl

# Higher concurrency, skip quality metrics
uv run python -m scripts.editing \
  --max-concurrent 20 \
  --no-quality \
  --input-path scratch/my_exp/inference_output.jsonl \
  --output-path scratch/my_exp/edited_dataset.jsonl

# Run multiple quality evaluations (code + LLM-judge)
uv run python -m scripts.editing \
  --quality-evaluations level_of_persona coherence \
  --quality-judge-provider openai \
  --quality-judge-model gpt-4o-mini \
  --input-path scratch/my_exp/inference_output.jsonl \
  --output-path scratch/my_exp/edited_dataset.jsonl

# Edit with a code-based editor
uv run python -m scripts.editing \
  --provider code \
  --code-editor scripts.editing.code_editors:reverse_text \
  --input-path scratch/my_exp/inference_output.jsonl \
  --output-path scratch/my_exp/edited_dataset.jsonl
```

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--provider` | `anthropic`, `openai`, or `code` | `anthropic` |
| `--model` | Model name | `claude-sonnet-4-20250514` |
| `--prompt-template` | Prompt template name | auto from `--persona` |
| `--code-editor` | Import path for code editor | `scripts.editing.code_editors:reverse_text` |
| `--max-concurrent` | Max concurrent API requests | `10` |
| `--timeout` | Request timeout (seconds) | `60` |
| `--input-path` | Input JSONL path (required) | — |
| `--output-path` | Output JSONL path | — |
| `--no-quality` | Disable quality metrics | off |
| `--quality-evaluations` | Evaluations for edit quality comparison | `level_of_persona` |
| `--quality-judge-provider` | Judge provider for LLM-based quality evals | `openai` |
| `--quality-judge-model` | Judge model for LLM-based quality evals | `gpt-4o-mini` |
| `--quality-judge-max-concurrent` | Max concurrent judge requests | `10` |

## Python Usage

```python
from pathlib import Path
from scripts.editing import run_editing, EditingConfig

config = EditingConfig(
    provider="anthropic",
    model="claude-sonnet-4-20250514",
    prompt_template="default_persona_shatter",
    output_path=Path("scratch/edited.jsonl"),
)
dataset, result = run_editing(config, input_path=Path("scratch/inference_output.jsonl"))
```

## Providers

- **`anthropic`**: Uses the Anthropic API. Requires `ANTHROPIC_API_KEY` env var.
- **`openai`**: Uses the OpenAI API. Requires `OPENAI_API_KEY` env var.
- **`code`**: Runs a local Python function `def edit(text: str, record: dict) -> str`.

## Quality Metrics

When enabled (default), edit quality is computed through the shared `scripts.evaluation` module.
For each evaluation metric key, editing stores:
- `<metric>.original` (on pre-edit response)
- `<metric>.edited` (on post-edit response)
- `<metric>.delta` (numeric metrics only)

The default quality evaluation is `level_of_persona`, resolved from the active persona in `scripts.common.persona_metrics` (for example, counting `"o"` characters for `o_avoiding`). Disable with `--no-quality`.

See [Evaluation README — Persona Registry](../evaluation/README.md#persona-registry) for the full list of available personas and how to select one via `--persona`.
