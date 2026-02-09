"""Inference stage configuration."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from scripts.common.config import DatasetConfig, GenerationConfig


class LocalProviderConfig(BaseModel):
    """Local model loading settings (HuggingFace transformers)."""

    dtype: str = "bfloat16"
    device_map: str = "auto"
    revision: str = "main"


class OpenAIProviderConfig(BaseModel):
    """OpenAI-compatible API settings.

    Works with OpenAI, OpenRouter, vLLM, and any OpenAI-compatible endpoint.
    """

    base_url: str | None = None  # None = use default OpenAI API
    api_key_env: str = "OPENAI_API_KEY"  # Environment variable name for API key


class InferenceConfig(BaseModel):
    """Configuration for the inference stage.

    Example:
        config = InferenceConfig(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            provider="local",
            dataset=DatasetConfig(
                source="huggingface",
                name="vicgalle/alpaca-gpt4",
                max_samples=10,
            ),
            generation=GenerationConfig(
                max_new_tokens=500,
                temperature=0.7,
            ),
            output_path=Path("scratch/inference_output.jsonl"),
        )
        result = run_inference(config)
    """

    # Model settings
    model: str = "meta-llama/Llama-3.1-8B-Instruct"
    provider: str = "local"  # "local" or "openai"

    # Dataset settings
    dataset: DatasetConfig = DatasetConfig()

    # Generation settings
    generation: GenerationConfig = GenerationConfig()

    # Provider-specific settings
    local: LocalProviderConfig = LocalProviderConfig()
    openai: OpenAIProviderConfig = OpenAIProviderConfig()

    # Output
    output_path: Path | None = None  # If None, returns dataset without saving


class InferenceResult(BaseModel):
    """Result from running inference."""

    class Config:
        arbitrary_types_allowed = True

    output_path: Path | None = None
    num_samples: int = 0
