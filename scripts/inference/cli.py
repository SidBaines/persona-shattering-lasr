#!/usr/bin/env python3
"""CLI entry point for the inference stage."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.common.config import DatasetConfig, GenerationConfig
from scripts.inference.config import (
    AnthropicProviderConfig,
    InferenceConfig,
    OpenAIBatchConfig,
    OpenAIProviderConfig,
    OpenRouterProviderConfig,
)
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
        choices=["local", "openai", "openrouter", "anthropic"],
        default="local",
        help="Inference provider: local, openai, openrouter, or anthropic",
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
        "--openai-base-url",
        type=str,
        default=None,
        help="Base URL for OpenAI API (default: official OpenAI endpoint)",
    )
    parser.add_argument(
        "--openai-api-key-env",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable name for OpenAI API key (default: OPENAI_API_KEY)",
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
    parser.add_argument(
        "--openai-min-output-tokens",
        type=int,
        default=256,
        help="Minimum max_output_tokens for OpenAI Responses (default: 256).",
    )
    parser.add_argument(
        "--openai-retry-max-output-tokens",
        type=int,
        default=1024,
        help="Retry cap for max_output_tokens when incomplete (default: 1024).",
    )
    parser.add_argument(
        "--openai-no-retry-on-incomplete",
        action="store_true",
        help="Disable retries when Responses API returns incomplete output.",
    )
    parser.add_argument(
        "--openai-batch",
        action="store_true",
        help="Use OpenAI Batch API (Responses endpoint).",
    )
    parser.add_argument(
        "--openai-batch-completion-window",
        type=str,
        default="24h",
        help="Batch completion window (default: 24h).",
    )
    parser.add_argument(
        "--openai-batch-poll-interval",
        type=int,
        default=10,
        help="Polling interval in seconds (default: 10).",
    )
    parser.add_argument(
        "--openai-batch-timeout",
        type=int,
        default=None,
        help="Optional timeout in seconds for batch completion.",
    )
    parser.add_argument(
        "--openai-batch-include-sampling",
        action="store_true",
        help="Include temperature/top_p in batch requests (default: omitted).",
    )
    parser.add_argument(
        "--openai-batch-run-dir",
        type=str,
        default=None,
        help="Run directory name under scratch/ for batch artifacts.",
    )
    parser.add_argument(
        "--openai-batch-resume",
        action="store_true",
        help="Resume an existing OpenAI batch using run-dir metadata.",
    )

    # OpenRouter provider settings
    parser.add_argument(
        "--openrouter-base-url",
        type=str,
        default="https://openrouter.ai/api/v1",
        help="Base URL for OpenRouter API (default: https://openrouter.ai/api/v1)",
    )
    parser.add_argument(
        "--openrouter-api-key-env",
        type=str,
        default="OPENROUTER_API_KEY",
        help="Environment variable name for OpenRouter API key (default: OPENROUTER_API_KEY)",
    )
    parser.add_argument(
        "--openrouter-app-url",
        type=str,
        default=None,
        help="Optional app URL for OpenRouter attribution",
    )
    parser.add_argument(
        "--openrouter-app-name",
        type=str,
        default=None,
        help="Optional app name for OpenRouter attribution",
    )

    # Anthropic provider settings
    parser.add_argument(
        "--anthropic-api-key-env",
        type=str,
        default="ANTHROPIC_API_KEY",
        help="Environment variable name for Anthropic API key (default: ANTHROPIC_API_KEY)",
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
            base_url=args.openai_base_url,
            api_key_env=args.openai_api_key_env,
            reasoning_effort=args.openai_reasoning_effort,
            verbosity=args.openai_verbosity,
            min_output_tokens=args.openai_min_output_tokens,
            retry_on_incomplete=not args.openai_no_retry_on_incomplete,
            retry_max_output_tokens=args.openai_retry_max_output_tokens,
            batch=OpenAIBatchConfig(
                enabled=args.openai_batch,
                completion_window=args.openai_batch_completion_window,
                poll_interval_seconds=args.openai_batch_poll_interval,
                timeout_seconds=args.openai_batch_timeout,
                include_sampling=args.openai_batch_include_sampling,
                run_dir=args.openai_batch_run_dir,
                resume=args.openai_batch_resume,
            ),
        ),
        openrouter=OpenRouterProviderConfig(
            base_url=args.openrouter_base_url,
            api_key_env=args.openrouter_api_key_env,
            app_url=args.openrouter_app_url,
            app_name=args.openrouter_app_name,
        ),
        anthropic=AnthropicProviderConfig(
            api_key_env=args.anthropic_api_key_env,
        ),
        output_path=Path(args.output_path) if args.output_path else None,
    )

    run_inference(config)


if __name__ == "__main__":
    main()
