# Test Training

This module is a deterministic training-pipeline debug path.

It does four things:

1. Runs local inference on `vicgalle/alpaca-gpt4` for 1000 samples by default.
2. Applies a code-based edit to every response using a cyclic vowel substitution:
   `a -> e -> i -> o -> u -> a`.
3. Evaluates outputs with two simple word-count metrics:
   `count_the` and `count_thi`.
4. Trains a LoRA adapter on the edited responses and uses the same metrics for
   training-time evaluation.

This is intentionally not a realistic persona pipeline. It is a controlled,
easy-to-measure SFT objective for debugging whether inference, editing,
evaluation, and training are wired correctly.

## Files

- `toy_cyclic_vowel_sft.py`: end-to-end experiment script

## Run

From the repo root:

```bash
UV_CACHE_DIR=.uv-cache \
uv run python scripts/experiments/test_training/toy_cyclic_vowel_sft.py
```

## Reuse Existing Inference

If you already have an inference JSONL with `question` and `response` fields,
skip stage 1:

```bash
UV_CACHE_DIR=.uv-cache \
uv run python scripts/experiments/test_training/toy_cyclic_vowel_sft.py \
  --inference-path scratch/your_run/inference_responses.jsonl
```

## Notes

- Default base model: `meta-llama/Llama-3.1-8B-Instruct`
- Default generation length: `128` new tokens
- Default metrics: `count_the`, `count_thi`
- Default LoRA targets are inherited from the shared training config, so this
  trains LoRA adapters on both attention and MLP projection modules.
