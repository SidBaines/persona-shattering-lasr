#!/usr/bin/env python3
"""Quick smoke test for inference providers with a short prompt.

Usage:
    uv run python experiments/test_inference_providers.py \
      --prompt "Say hello in one sentence." \
      --openai-model gpt-5-nano-2025-08-07 \
      --openrouter-model meta-llama/Llama-3.1-8B-Instruct \
      --anthropic-model claude-3-5-sonnet-20241022

    # Include local provider
    uv run python experiments/test_inference_providers.py \
      --providers local openai \
      --local-model Qwen/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from scripts.common.config import GenerationConfig
from scripts.inference import (
    AnthropicProviderConfig,
    InferenceConfig,
    LocalProviderConfig,
    OpenAIProviderConfig,
    OpenRouterProviderConfig,
)
from scripts.inference.providers import get_provider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test inference providers with a single prompt.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Say hello in one sentence.",
        help="Prompt to send to each provider.",
    )
    parser.add_argument(
        "--providers",
        type=str,
        nargs="+",
        default=["openai", "openrouter", "anthropic"],
        choices=["local", "openai", "openrouter", "anthropic"],
        help="Providers to test (default: openai openrouter anthropic)",
    )

    parser.add_argument(
        "--local-model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Local HuggingFace model name.",
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default="gpt-5-nano-2025-08-07",
        help="OpenAI model name.",
    )
    parser.add_argument(
        "--openrouter-model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="OpenRouter model name.",
    )
    parser.add_argument(
        "--anthropic-model",
        type=str,
        default="claude-3-5-sonnet-20241022",
        help="Anthropic model name.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Max new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )

    return parser.parse_args()


def _build_config(provider: str, args: argparse.Namespace) -> InferenceConfig:
    if provider == "local":
        model = args.local_model
    elif provider == "openai":
        model = args.openai_model
    elif provider == "openrouter":
        model = args.openrouter_model
    elif provider == "anthropic":
        model = args.anthropic_model
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return InferenceConfig(
        model=model,
        provider=provider,
        generation=GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            batch_size=1,
        ),
        local=LocalProviderConfig(),
        openai=OpenAIProviderConfig(),
        openrouter=OpenRouterProviderConfig(),
        anthropic=AnthropicProviderConfig(),
    )


def main() -> None:
    load_dotenv()
    args = parse_args()

    print("=" * 60)
    print("INFERENCE PROVIDER SMOKE TEST")
    print("=" * 60)
    print(f"Prompt: {args.prompt}")
    print(f"Providers: {', '.join(args.providers)}")
    print("".rjust(60, "-"))

    for provider_name in args.providers:
        print(f"\n[{provider_name}]")
        try:
            config = _build_config(provider_name, args)
            provider = get_provider(provider_name, config)
            response = provider.generate(args.prompt)
            print(f"Model: {config.model}")
            print("Response:")
            if response.strip():
                print(response)
            else:
                print("[EMPTY RESPONSE] (check logs for provider details)")
        except Exception as exc:
            print(f"Error: {exc}")


if __name__ == "__main__":
    main()
