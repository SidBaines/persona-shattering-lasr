# Inference

Run LLM inference on a dataset, producing a response for each question. Supports local HuggingFace models and OpenAI-compatible APIs (OpenRouter, vLLM, etc.).

## CLI Usage

```bash
# Local model inference
uv run python -m scripts.inference \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset-name vicgalle/alpaca-gpt4 \
  --max-samples 100 \
  --output-path scratch/inference_output.jsonl

# OpenAI-compatible API (e.g., OpenRouter)
uv run python -m scripts.inference \
  --provider openai \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --base-url https://openrouter.ai/api/v1 \
  --api-key-env OPENROUTER_API_KEY \
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
| `--provider` | `local` or `openai` | `local` |
| `--dataset-name` | HuggingFace dataset name | — |
| `--dataset-source` | `huggingface` or `local` | `huggingface` |
| `--max-samples` | Limit number of samples | all |
| `--max-new-tokens` | Max tokens to generate | `512` |
| `--temperature` | Sampling temperature | `0.7` |
| `--batch-size` | Batch size | `8` |
| `--num-responses` | Responses per prompt | `1` |
| `--output-path` | Output JSONL path | — |
| `--base-url` | OpenAI-compatible API base URL | — |
| `--api-key-env` | Env var name for API key | `OPENAI_API_KEY` |

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
- **`openai`**: Calls any OpenAI-compatible API. Configure with `--base-url` and `--api-key-env`. Works with OpenAI, OpenRouter, vLLM, and similar endpoints.
