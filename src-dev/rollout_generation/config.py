"""Configuration models for long-context rollout generation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from scripts.common.config import DatasetConfig, GenerationConfig
from scripts.inference.config import (
    AnthropicProviderConfig,
    InferenceConfig,
    LocalProviderConfig,
    OpenAIProviderConfig,
    OpenRouterProviderConfig,
    RetryConfig,
)


class UserSimulatorConfig(BaseModel):
    """Configuration for generating the next user turn with a strong LLM."""

    provider: str = "openai"
    model: str = "gpt-5-nano-2025-08-07"
    prompt_template: str = "typical_user"
    prompt_format: Literal["chat_messages", "single_turn_text"] = "single_turn_text"
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    max_concurrent: int = 16
    timeout: int = 60
    retry: RetryConfig = Field(default_factory=RetryConfig)
    local: LocalProviderConfig = Field(default_factory=LocalProviderConfig)
    openai: OpenAIProviderConfig = Field(default_factory=OpenAIProviderConfig)
    openrouter: OpenRouterProviderConfig = Field(
        default_factory=OpenRouterProviderConfig
    )
    anthropic: AnthropicProviderConfig = Field(default_factory=AnthropicProviderConfig)


class ContextPolicyConfig(BaseModel):
    """Context-window policy for rollout prompting."""

    mode: Literal["full_history", "token_budget"] = "full_history"
    assistant_max_context_tokens: int | None = None
    user_max_context_tokens: int | None = None


class FailurePolicyConfig(BaseModel):
    """Per-turn retry limits before marking a sample terminal."""

    assistant_max_attempts_per_turn: int = 3
    user_max_attempts_per_turn: int = 3


class RolloutGenerationConfig(BaseModel):
    """Configuration for assistant<->user long-context rollout generation."""

    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    run_dir: Path
    num_assistant_turns: int
    num_rollouts_per_prompt: int = 1
    system_prompt: str | None = None

    assistant_inference: InferenceConfig
    user_simulator: UserSimulatorConfig = Field(default_factory=UserSimulatorConfig)

    transcript_variant: str = "rollout_base"
    context_policy: ContextPolicyConfig = Field(default_factory=ContextPolicyConfig)
    failure_policy: FailurePolicyConfig = Field(default_factory=FailurePolicyConfig)

    skip_final_user_turn: bool = True
    resume: bool = True
    overwrite_output: bool = False


class RolloutGenerationResult(BaseModel):
    """Result metadata for rollout generation."""

    output_path: Path | None = None
    num_conversations: int = 0
    num_completed: int = 0
    num_failed: int = 0
    num_assistant_turns_target: int = 0
    num_assistant_turns_completed: int = 0
    exports: dict[str, str] = Field(default_factory=dict)
