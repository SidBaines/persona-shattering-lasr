# Inference

Run LLM inference on a dataset, producing a response for each question. Supports local HuggingFace models plus OpenAI, OpenRouter, and Anthropic APIs.

## CLI Usage

```bash
# Local model inference
uv run python -m scripts.inference \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset-name vicgalle/alpaca-gpt4 \
  --max-samples 100 \
  --output-path scratch/inference_output.jsonl

# OpenAI API
uv run python -m scripts.inference \
  --provider openai \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --openai-api-key-env OPENAI_API_KEY \
  --dataset-name vicgalle/alpaca-gpt4 \
  --max-samples 50 \
  --output-path scratch/inference_output.jsonl

# OpenAI Batch API (Responses endpoint)
uv run python -m scripts.inference \
  --provider openai \
  --openai-batch \
  --model gpt-5-nano-2025-08-07 \
  --openai-api-key-env OPENAI_API_KEY \
  --dataset-name vicgalle/alpaca-gpt4 \
  --max-samples 200 \
  --output-path scratch/inference_output.jsonl

Note: Batch runs are asynchronous and will poll until completion (default 24h window).

# OpenRouter API
uv run python -m scripts.inference \
  --provider openrouter \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --openrouter-api-key-env OPENROUTER_API_KEY \
  --dataset-name vicgalle/alpaca-gpt4 \
  --max-samples 50 \
  --output-path scratch/inference_output.jsonl

# Anthropic API
uv run python -m scripts.inference \
  --provider anthropic \
  --model claude-3-5-sonnet-20241022 \
  --anthropic-api-key-env ANTHROPIC_API_KEY \
  --dataset-name vicgalle/alpaca-gpt4 \
  --max-samples 50 \
  --output-path scratch/inference_output.jsonl

# Multiple responses per prompt
uv run python -m scripts.inference \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --num-responses 3 \
  --temperature 0.9 \
  --dataset-name vicgalle/alpaca-gpt4 \
  --output-path scratch/multi_response.jsonl
```

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | Model name or HuggingFace path | `meta-llama/Llama-3.1-8B-Instruct` |
| `--provider` | `local`, `openai`, `openrouter`, or `anthropic` | `local` |
| `--dataset-name` | HuggingFace dataset name | — |
| `--dataset-source` | `huggingface` or `local` | `huggingface` |
| `--max-samples` | Limit number of samples | all |
| `--max-new-tokens` | Max tokens to generate | `512` |
| `--temperature` | Sampling temperature | `0.7` |
| `--batch-size` | Batch size | `8` |
| `--num-responses` | Responses per prompt | `1` |
| `--output-path` | Output JSONL path | — |
| `--openai-base-url` | OpenAI API base URL | — |
| `--openai-api-key-env` | Env var name for OpenAI API key | `OPENAI_API_KEY` |
| `--openai-batch` | Use OpenAI Batch API | `false` |
| `--openai-batch-completion-window` | Batch completion window | `24h` |
| `--openai-batch-poll-interval` | Batch poll interval (seconds) | `10` |
| `--openai-batch-timeout` | Batch timeout (seconds) | — |
| `--openai-batch-include-sampling` | Include temperature/top_p in batch | `false` |
| `--openrouter-base-url` | OpenRouter API base URL | `https://openrouter.ai/api/v1` |
| `--openrouter-api-key-env` | Env var name for OpenRouter API key | `OPENROUTER_API_KEY` |
| `--openrouter-app-url` | Optional OpenRouter app URL | — |
| `--openrouter-app-name` | Optional OpenRouter app name | — |
| `--anthropic-api-key-env` | Env var name for Anthropic API key | `ANTHROPIC_API_KEY` |

## Python Usage

```python
from pathlib import Path
from scripts.inference import run_inference, InferenceConfig
from scripts.common.config import DatasetConfig, GenerationConfig

config = InferenceConfig(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    provider="local",
    dataset=DatasetConfig(
        source="huggingface",
        name="vicgalle/alpaca-gpt4",
        max_samples=10,
    ),
    generation=GenerationConfig(max_new_tokens=500),
    output_path=Path("scratch/output.jsonl"),
)
dataset, result = run_inference(config)
```

## Providers

- **`local`**: Loads model via HuggingFace `transformers`. Runs on local GPU.
- **`openai`**: Calls the OpenAI API.
- **`openrouter`**: Calls the OpenRouter API (OpenAI-compatible).
- **`anthropic`**: Calls the Anthropic API.
