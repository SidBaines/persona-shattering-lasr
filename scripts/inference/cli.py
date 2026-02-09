#!/usr/bin/env python3
"""CLI entry point for the inference stage."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.common.config import DatasetConfig, GenerationConfig
from scripts.inference.config import InferenceConfig, OpenAIProviderConfig
from scripts.inference.run import run_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LLM inference on a dataset.",
    )

    # Model settings
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name or HuggingFace path (default: meta-llama/Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["local", "openai"],
        default="local",
        help="Inference provider: 'local' (HuggingFace) or 'openai' (OpenAI-compatible API)",
    )

    # Dataset settings
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="HuggingFace dataset name (e.g., vicgalle/alpaca-gpt4)",
    )
    parser.add_argument(
        "--dataset-source",
        type=str,
        choices=["huggingface", "local"],
        default="huggingface",
        help="Dataset source type (default: huggingface)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process",
    )

    # Generation settings
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference (default: 8)",
    )
    parser.add_argument(
        "--num-responses",
        type=int,
        default=1,
        help="Number of responses per prompt (default: 1)",
    )

    # Output
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save output JSONL file",
    )

    # OpenAI provider settings
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL for OpenAI-compatible API (e.g., OpenRouter, vLLM)",
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable name for API key (default: OPENAI_API_KEY)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = InferenceConfig(
        model=args.model,
        provider=args.provider,
        dataset=DatasetConfig(
            source=args.dataset_source,
            name=args.dataset_name,
            max_samples=args.max_samples,
        ),
        generation=GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            batch_size=args.batch_size,
            num_responses_per_prompt=args.num_responses,
        ),
        openai=OpenAIProviderConfig(
            base_url=args.base_url,
            api_key_env=args.api_key_env,
        ),
        output_path=Path(args.output_path) if args.output_path else None,
    )

    run_inference(config)


if __name__ == "__main__":
    main()
