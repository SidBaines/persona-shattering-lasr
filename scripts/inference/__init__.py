"""Inference module for running LLM inference on datasets.

Example:
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
"""

from scripts.inference.config import (
    AnthropicProviderConfig,
    InferenceConfig,
    InferenceResult,
    LocalProviderConfig,
    OpenAIBatchConfig,
    OpenAIProviderConfig,
    OpenRouterProviderConfig,
    RetryConfig,
)
from scripts.inference.run import run_inference, run_inference_async
from scripts.inference.cli import main
from scripts.inference.providers import get_provider
from scripts.inference.providers.base import (
    InferenceProvider,
    StructuredGenerationResult,
    StructuredOutputSpec,
)

__all__ = [
    # Config classes
    "InferenceConfig",
    "InferenceResult",
    "LocalProviderConfig",
    "OpenAIBatchConfig",
    "OpenAIProviderConfig",
    "OpenRouterProviderConfig",
    "AnthropicProviderConfig",
    "RetryConfig",
    # Run function
    "run_inference",
    "run_inference_async",
    # CLI entry point
    "main",
    # Providers
    "get_provider",
    "InferenceProvider",
    "StructuredOutputSpec",
    "StructuredGenerationResult",
]
