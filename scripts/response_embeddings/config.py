"""Configuration models for response embedding extraction and variance analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class LocalHFEmbeddingConfig(BaseModel):
    """Local HuggingFace embedding model settings."""

    model: str = "Qwen/Qwen3-Embedding-4B"
    revision: str = "main"
    dtype: str = "bfloat16"
    device_map: str = "auto"
    trust_remote_code: bool = True
    max_length: int = 2048
    batch_size: int = 8
    normalize: bool = True


class OpenAIEmbeddingConfig(BaseModel):
    """OpenAI embedding model settings."""

    model: str = "text-embedding-3-small"
    api_key_env: str = "OPENAI_API_KEY"
    dimensions: int | None = None
    batch_size: int = 128
    normalize: bool = True
    max_retries: int = 6
    initial_backoff_seconds: float = 2.0
    max_backoff_seconds: float = 60.0


class ResponseEmbeddingConfig(BaseModel):
    """Configuration for assistant-response embedding extraction."""

    run_dir: Path
    analysis_unit: Literal[
        "assistant_all_turns",
        "assistant_final_turn",
        "assistant_first_turn",
    ] = "assistant_all_turns"
    target_variant: str | None = None

    backend: Literal["local_hf", "openai"] = "local_hf"
    local_hf: LocalHFEmbeddingConfig = Field(default_factory=LocalHFEmbeddingConfig)
    openai: OpenAIEmbeddingConfig = Field(default_factory=OpenAIEmbeddingConfig)

    artifact_slug: str | None = None
    output_prefix: str = "response_embeddings"
    resume: bool = True
    overwrite_output: bool = False


class ResponseEmbeddingResult(BaseModel):
    """Result metadata for response embeddings stage."""

    artifact_slug: str | None = None
    artifact_dir: Path | None = None
    metadata_path: Path | None = None
    embeddings_path: Path | None = None
    variance_path: Path | None = None
    manifest_path: Path | None = None

    num_samples: int = 0
    embedding_dim: int = 0
    analysis_unit: str = "assistant_all_turns"
