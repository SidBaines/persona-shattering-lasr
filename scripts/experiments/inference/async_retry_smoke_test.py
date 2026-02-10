#!/usr/bin/env python3
"""Async retry/concurrency smoke test for inference providers."""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from scripts.common.config import GenerationConfig
from scripts.inference import (
    AnthropicProviderConfig,
    InferenceConfig,
    LocalProviderConfig,
    OpenAIProviderConfig,
    OpenRouterProviderConfig,
    RetryConfig,
    run_inference_async,
)
from scripts.utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Async smoke test for inference retries and concurrency.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["local", "openai", "openrouter", "anthropic"],
        default="openai",
        help="Inference provider to test.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-nano-2025-08-07",
        help="Model name for the selected provider.",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="Write a one-sentence summary about test item {i}.",
        help="Prompt template. Use {i} to inject the prompt index.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=8,
        help="Number of prompts to send.",
    )
    parser.add_argument(
        "--num-responses",
        type=int,
        default=1,
        help="Number of responses per prompt.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference (controls prompt chunking).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100000,
        help="Maximum new/output tokens.",
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
        help="Top-p for sampling.",
    )

    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum concurrent API requests.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Request timeout in seconds (0 disables).",
    )
    parser.add_argument(
        "--retry-max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts.",
    )
    parser.add_argument(
        "--retry-backoff-factor",
        type=float,
        default=2.0,
        help="Exponential backoff factor.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first error instead of continuing.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Log level (DEBUG, INFO, WARNING, ERROR).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional output JSONL path.",
    )

    parser.add_argument(
        "--openai-reasoning-effort",
        type=str,
        choices=["none", "low", "medium", "high"],
        default=None,
        help="OpenAI reasoning effort (optional).",
    )
    parser.add_argument(
        "--openai-verbosity",
        type=str,
        choices=["low", "medium", "high"],
        default=None,
        help="OpenAI verbosity (optional).",
    )

    return parser.parse_args()


def _build_dataset(prompt_template: str, num_prompts: int) -> Dataset:
    prompts: list[str] = []
    for i in range(num_prompts):
        if "{i}" in prompt_template:
            prompt = prompt_template.format(i=i)
        else:
            prompt = f"{prompt_template} (item {i})"
        prompts.append(prompt)
    return Dataset.from_list([{"question": prompt} for prompt in prompts])


async def _run_async(args: argparse.Namespace) -> None:
    dataset = _build_dataset(args.prompt_template, args.num_prompts)

    timeout = None if args.timeout <= 0 else args.timeout
    config = InferenceConfig(
        model=args.model,
        provider=args.provider,
        generation=GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            batch_size=args.batch_size,
            num_responses_per_prompt=args.num_responses,
        ),
        max_concurrent=args.max_concurrent,
        timeout=timeout,
        retry=RetryConfig(
            max_retries=args.retry_max_retries,
            backoff_factor=args.retry_backoff_factor,
        ),
        continue_on_error=not args.fail_fast,
        local=LocalProviderConfig(),
        openai=OpenAIProviderConfig(
            reasoning_effort=args.openai_reasoning_effort,
            verbosity=args.openai_verbosity,
        ),
        openrouter=OpenRouterProviderConfig(),
        anthropic=AnthropicProviderConfig(),
        output_path=Path(args.output_path) if args.output_path else None,
    )

    start = time.perf_counter()
    result_dataset, result = await run_inference_async(config, dataset)
    elapsed = time.perf_counter() - start

    responses = result_dataset["response"]
    empty = sum(1 for text in responses if not (text or "").strip())

    print("=" * 60)
    print("ASYNC INFERENCE SMOKE TEST")
    print("=" * 60)
    print(f"Provider: {args.provider}")
    print(f"Model: {args.model}")
    print(f"Prompts: {args.num_prompts}")
    print(f"Responses per prompt: {args.num_responses}")
    print(f"Max concurrent: {args.max_concurrent}")
    print(f"Retries: {args.retry_max_retries} (backoff {args.retry_backoff_factor})")
    print(f"Timeout: {timeout}")
    print(f"Continue on error: {not args.fail_fast}")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Total responses: {len(responses)}")
    print(f"Empty responses: {empty}")
    print("=" * 60)

    if responses:
        print("Sample responses:")
        for i, text in enumerate(responses[: min(3, len(responses))]):
            preview = (text or "").strip()
            if len(preview) > 200:
                preview = preview[:200] + "..."
            print(f"[{i}] {preview}")


def main() -> None:
    load_dotenv()
    args = parse_args()
    setup_logging(args.log_level)

    asyncio.run(_run_async(args))


if __name__ == "__main__":
    main()
