# Editing

Edit model responses using an LLM API. Sends each response through a prompt template (e.g., passive voice conversion) and collects edited outputs with optional quality metrics.

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
```

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--provider` | `anthropic` or `openai` | `anthropic` |
| `--model` | Model name | `claude-sonnet-4-20250514` |
| `--prompt-template` | Prompt template name | `default_persona_shatter` |
| `--max-concurrent` | Max concurrent API requests | `10` |
| `--timeout` | Request timeout (seconds) | `60` |
| `--input-path` | Input JSONL path (required) | — |
| `--output-path` | Output JSONL path | — |
| `--no-quality` | Disable quality metrics | off |

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

## Quality Metrics

When enabled (default), quality metrics are computed for each edited response and included in the output JSONL. The `count_passive` metric tracks passive voice construction usage. Disable with `--no-quality`.
