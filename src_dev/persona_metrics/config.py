"""Evaluation stage configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from src_dev.common.config import DatasetConfig


class JudgeLLMConfig(BaseModel):
    """Configuration for LLM-as-judge evaluations (e.g., CoherenceEvaluation).

    Default model is Gemini Flash via OpenRouter — cheapest option with
    good calibration (QWK 0.888 on coherence, 0.93+ on OCEAN traits).

    See :data:`JUDGE_PANEL` for the recommended model panel and
    :func:`judge_config` for a convenience constructor.
    """

    provider: str = "openrouter"
    model: str = "google/gemini-2.0-flash-001"
    api_key_env: str | None = None  # If None, uses default for provider
    max_tokens: int = 1024
    temperature: float = 0.0  # Deterministic by default for judging
    max_concurrent: int = 10
    timeout: int = 60
    max_retries: int = 3  # Number of retries on transient errors (e.g. rate limits)
    backoff_factor: float = 2.0  # Exponential backoff multiplier between retries


class PersonaMetricSpec(BaseModel):
    """Specification for a single evaluation with optional per-evaluation params.

    Use this when you need to pass evaluation-specific configuration such as
    custom thresholds, prompt templates, or a different judge model.

    Example:
        PersonaMetricSpec(name="coherence", params={"judge_config": JudgeLLMConfig(model="gpt-4o")})
        PersonaMetricSpec(name="regex_match", params={"pattern": r"\\b(safe|unsafe)\\b"})
    """

    name: str
    params: dict[str, Any] = {}


class PersonaMetricsConfig(BaseModel):
    """Configuration for the evaluation stage.

    Example:
        config = PersonaMetricsConfig(
            evaluations=["count_o", "coherence"],
            dataset=DatasetConfig(source="local", path="scratch/inference_output.jsonl"),
            response_column="response",
            question_column="question",
            output_path=Path("scratch/evaluation_results.jsonl"),
        )
        dataset, result = run_persona_metrics(config)

        # With per-evaluation params:
        config = PersonaMetricsConfig(
            evaluations=[
                "count_o",
                PersonaMetricSpec(name="coherence", params={"judge_config": JudgeLLMConfig(model="gpt-4o")}),
            ],
        )
    """

    # Which evaluations to run (strings or PersonaMetricSpec for per-eval config)
    evaluations: list[str | PersonaMetricSpec] = ["count_o"]

    # Dataset settings (for standalone run; not needed when called inline)
    dataset: DatasetConfig = DatasetConfig()
    run_dir: Path | None = None  # Canonical run directory under scratch/runs/<run_id>
    target_variant: str | None = None  # If set, evaluate edited variant instead of base inference

    # Column mapping
    response_column: str = "response"
    question_column: str | None = "question"

    # LLM judge settings (for evaluations that need an LLM)
    judge: JudgeLLMConfig = JudgeLLMConfig()

    # Key under which per-record metrics are embedded in the dataset
    metrics_key: str = "persona_metrics"

    # Output
    output_path: Path | None = None


class PersonaMetricsResult(BaseModel):
    """Result from running evaluation."""

    class Config:
        arbitrary_types_allowed = True

    output_path: Path | None = None
    num_samples: int = 0
    evaluations_run: list[str] = []
    aggregates: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Recommended judge panel (calibrated April 2026)
# ---------------------------------------------------------------------------
#
# Calibrated on golden datasets in data/judge_calibration/ for all 5 OCEAN
# traits (-4..+4) and coherence (0..10).  Calibration script:
#   scripts_dev/persona_metrics/llm_judge/golden_calibration.py
#
# Judge definitions (prompts, few-shot examples) live in:
#   src_dev/persona_metrics/metrics/ocean_v2.py    — OCEAN traits
#   src_dev/persona_metrics/metrics/coherence.py   — Coherence (CoherenceV2Evaluation)
#   src_dev/common/persona_definitions.py          — OCEAN trait definitions
#   src_dev/common/coherence_definition.py         — Coherence dimension definitions
#
# Golden datasets:
#   data/judge_calibration/{trait}.jsonl  — 33-36 items per trait, hand-labeled
#
# Selection criteria (coherence QWK / OCEAN mean Spearman):
#   Gemini Flash:  good quality, cheapest, highest throughput — default
#   Haiku 3.5:    best quality + throughput balance
#   Kimi K2:      best quality, BUT heavily rate-limited on OpenRouter
#                 (50 rpm, often 429s even at concurrency=3; use for
#                 small batches or when quality matters most)
#   DeepSeek V3:  good quality, cheap fallback
#
# Temperature: 0.0 for production judging (deterministic).
# Use 0.7 only for calibration runs measuring self-consistency.
#
# Retired:
#   GPT-5 Mini:   worst calibration of all tested, most expensive, Azure 403
#                 content-policy blocks on personality prompts via OpenRouter
#   GPT-4o Mini:  superseded by calibrated panel
#   Llama 4 Scout: poor rank-ordering (Spearman 0.86)

JUDGE_PANEL: dict[str, JudgeLLMConfig] = {
    "gemini_flash": JudgeLLMConfig(
        provider="openrouter",
        model="google/gemini-2.0-flash-001",
        max_concurrent=15,
    ),
    "haiku": JudgeLLMConfig(
        provider="openrouter",
        model="anthropic/claude-3.5-haiku",
        max_concurrent=15,
    ),
    "kimi_k2": JudgeLLMConfig(
        provider="openrouter",
        model="moonshotai/kimi-k2",
        max_concurrent=3,  # 50 rpm rate limit on OpenRouter; 429s common even at 5
    ),
    "deepseek_v3": JudgeLLMConfig(
        provider="openrouter",
        model="deepseek/deepseek-chat-v3",
        max_concurrent=15,
    ),
}


def judge_config(
    name: str = "gemini_flash",
    *,
    temperature: float = 0.0,
    timeout: int = 60,
) -> JudgeLLMConfig:
    """Return a judge config from the recommended panel.

    Args:
        name: Panel key — one of "gemini_flash", "haiku", "kimi_k2", "deepseek_v3".
        temperature: Override temperature (default 0.0 for deterministic judging,
            use 0.7 for self-consistency measurement in calibration).
        timeout: Request timeout in seconds.

    Returns:
        A :class:`JudgeLLMConfig` ready to pass to any judge metric.

    Example::

        from src_dev.persona_metrics.config import judge_config
        cfg = judge_config("haiku")
        cfg = judge_config("kimi_k2", temperature=0.7)
    """
    if name not in JUDGE_PANEL:
        available = ", ".join(sorted(JUDGE_PANEL))
        raise KeyError(f"Unknown judge '{name}'. Available: {available}")
    return JUDGE_PANEL[name].model_copy(update={"temperature": temperature, "timeout": timeout})
