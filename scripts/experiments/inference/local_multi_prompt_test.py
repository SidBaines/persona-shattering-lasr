#!/usr/bin/env python3
"""Quick local inference sanity check with hardcoded prompts.

Sends N hardcoded prompts to the local inference provider and requests M
responses per prompt.

Usage:
    uv run python scripts/experiments/inference/local_multi_prompt_test.py
    uv run python scripts/experiments/inference/local_multi_prompt_test.py --num-responses 3
    uv run python scripts/experiments/inference/local_multi_prompt_test.py --model Qwen/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

# Keep HF-related caches in /workspace where there is more disk capacity.
os.environ["XDG_CACHE_HOME"] = "/workspace/.cache"
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/workspace/.cache/huggingface/hub"
os.environ["HF_DATASETS_CACHE"] = "/workspace/.cache/huggingface/datasets"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/.cache/huggingface/transformers"

from scripts.common.config import GenerationConfig
from scripts.inference import InferenceConfig, LocalProviderConfig, run_inference

# N is determined by length of this list. Start with N=2 prompts.
PROMPTS: list[str] = [
    "What are three practical ways to improve focus when working from home?",
    "Explain why tides happen, in simple language for a 12-year-old.",
]

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_NUM_RESPONSES = 3  # M=3


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a quick local inference test with hardcoded prompts.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Local HuggingFace model name.",
    )
    parser.add_argument(
        "--num-responses",
        type=int,
        default=DEFAULT_NUM_RESPONSES,
        help="Number of responses per prompt (M).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=10000,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run id (default: auto timestamp).",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the local inference sanity test."""
    args = _parse_args()

    if args.num_responses < 1:
        raise ValueError("--num-responses must be >= 1")

    for cache_dir in (
        "/workspace/.cache",
        "/workspace/.cache/huggingface",
        "/workspace/.cache/huggingface/hub",
        "/workspace/.cache/huggingface/datasets",
        "/workspace/.cache/huggingface/transformers",
    ):
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    load_dotenv()

    run_id = args.run_id or f"local-inference-test-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    output_dir = Path("scratch") / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = Dataset.from_list([{"question": prompt} for prompt in PROMPTS])

    config = InferenceConfig(
        model=args.model,
        provider="local",
        generation=GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            batch_size=args.batch_size,
            num_responses_per_prompt=args.num_responses,
        ),
        local=LocalProviderConfig(),
        output_path=output_dir / "inference_output.jsonl",
    )

    result_dataset, result = run_inference(config, dataset)

    print("=" * 60)
    print("LOCAL INFERENCE TEST")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Prompts (N): {len(PROMPTS)}")
    print(f"Responses per prompt (M): {args.num_responses}")
    print(f"Total rows: {result.num_samples}")
    print(f"Failures: {result.num_failed}")
    print(f"Output: {result.output_path}")
    print("=" * 60)

    rows = result_dataset.to_list()
    for prompt_index, prompt in enumerate(PROMPTS):
        print(f"\nPrompt {prompt_index + 1}: {prompt}")
        prompt_rows = [row for row in rows if row["question"] == prompt]
        prompt_rows.sort(key=lambda row: row.get("response_index", 0))
        for row in prompt_rows:
            response_index = row.get("response_index", 0)
            print(f"  [{response_index}] {row.get('response', '').strip()}")


if __name__ == "__main__":
    main()
