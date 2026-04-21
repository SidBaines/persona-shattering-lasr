"""Configuration for frustration evaluation runs."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from src_dev.common.config import GenerationConfig
from src_dev.inference.config import (
    AnthropicProviderConfig,
    InferenceConfig,
    LocalProviderConfig,
    OpenAIProviderConfig,
    OpenRouterProviderConfig,
    RetryConfig,
    VllmProviderConfig,
)
from src_dev.persona_metrics.config import JudgeLLMConfig


class FrustrationEvalConfig(BaseModel):
    """Top-level configuration for a frustration evaluation run."""

    model_config = {"arbitrary_types_allowed": True}

    # ── Target model ──────────────────────────────────────────────────────
    model: str = "google/gemma-3-27b-it"
    provider: Literal["local", "vllm", "openai", "openrouter", "anthropic"] = "openrouter"
    generation: GenerationConfig = Field(
        default_factory=lambda: GenerationConfig(
            temperature=1.0,  # paper always uses temperature=1
            max_new_tokens=2048,
            top_p=1.0,
            do_sample=True,
        )
    )

    # Provider-specific configs
    local: LocalProviderConfig = Field(default_factory=LocalProviderConfig)
    vllm: VllmProviderConfig = Field(default_factory=VllmProviderConfig)
    openai: OpenAIProviderConfig = Field(default_factory=OpenAIProviderConfig)
    openrouter: OpenRouterProviderConfig = Field(default_factory=OpenRouterProviderConfig)
    anthropic: AnthropicProviderConfig = Field(default_factory=AnthropicProviderConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)

    # ── Evaluation categories ─────────────────────────────────────────────
    # Names of categories to run (from prompts.py). If empty, runs all main categories.
    categories: list[str] = Field(default_factory=list)

    # ── Judge ─────────────────────────────────────────────────────────────
    judge: JudgeLLMConfig = Field(
        default_factory=lambda: JudgeLLMConfig(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            temperature=0.0,
            max_tokens=512,
            max_concurrent=16,
            max_retries=3,
        )
    )

    # ── Execution ─────────────────────────────────────────────────────────
    max_concurrent: int = 16  # concurrent rollouts
    timeout: int | None = 600  # per-request HTTP timeout (seconds); long gens need >60
    output_dir: Path = Path("scratch/evals/frustration_eval")
    run_name: str = ""  # auto-generated from model name if empty
    seed: int = 42
    # If True, score responses per-turn (not just final turn)
    score_all_turns: bool = True

    # CLI overrides applied to all categories at runtime
    override_num_turns: int | None = None
    override_num_rollouts: int | None = None
    override_num_prompts: int | None = None

    def to_inference_config(self) -> InferenceConfig:
        """Build an InferenceConfig for the target model."""
        return InferenceConfig(
            model=self.model,
            provider=self.provider,
            generation=self.generation,
            max_concurrent=self.max_concurrent,
            timeout=self.timeout,
            local=self.local,
            vllm=self.vllm,
            openai=self.openai,
            openrouter=self.openrouter,
            anthropic=self.anthropic,
            retry=self.retry,
        )
