"""Configuration models for symmetric self-chat generation."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from scripts.common.config import DatasetConfig
from scripts.inference.config import InferenceConfig


class HfUploadConfig(BaseModel):
    """Optional Hugging Face dataset upload settings."""

    enabled: bool = False
    repo_id: str | None = None
    path_in_repo: str = "runs"
    commit_message: str = "Upload self-chat generation run"


class SelfChatGenerationConfig(BaseModel):
    """Configuration for symmetric self-chat generation."""

    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    run_dir: Path
    num_generated_turns: int
    num_rollouts_per_prompt: int = 1
    system_prompt: str | None = None

    speaker_a_inference: InferenceConfig
    speaker_b_inference: InferenceConfig | None = None

    resume: bool = True
    overwrite_output: bool = False
    hf_upload: HfUploadConfig = Field(default_factory=HfUploadConfig)


class SelfChatGenerationResult(BaseModel):
    """Result metadata for self-chat generation."""

    output_path: Path | None = None
    num_conversations: int = 0
    num_completed: int = 0
    num_failed: int = 0
    num_generated_turns_target: int = 0
    num_generated_turns_completed: int = 0
    exports: dict[str, str] = Field(default_factory=dict)
    hf_dataset_url: str | None = None
