"""Minimal async LLM-judge primitive for the persona-jailbreak eval.

The existing :class:`src_dev.persona_metrics.metrics.llm_judge_base.LLMJudgeMetric`
is integer-score-oriented and fixes the prompt template at two placeholders
(``{question_text}``, ``{response}``). The paper rubric (Appendix D.2.2) wants
four placeholders (``{request}``, ``{response}``, ``{behavior}``, ``{action}``)
and produces a 9-class string enum, so we'd be fighting the abstraction.

This module is a thin async wrapper around ``InferenceConfig`` + ``get_provider``
that lets a subclass pin its own prompt template and parser. ~80 lines, no
Inspect, no scale-sweep harness — just "give me N input dicts, return N
parsed JSON results with retries and concurrency."
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping

from src_dev.common.config import GenerationConfig
from src_dev.inference.config import (
    AnthropicProviderConfig,
    InferenceConfig,
    OpenAIProviderConfig,
    OpenRouterProviderConfig,
    RetryConfig,
)
from src_dev.inference.providers import get_provider
from src_dev.inference.providers.base import InferenceProvider
from src_dev.persona_metrics.config import JudgeLLMConfig

logger = logging.getLogger(__name__)


@dataclass
class JudgeOutcome:
    """One judge call's structured result.

    ``label`` is the canonical category string the rubric defines. ``raw_text``
    is the judge's full reply (kept for debugging). ``parse_error`` is non-None
    when the judge's reply couldn't be parsed.
    """

    label: str | None
    analysis: str | None
    raw_text: str
    parse_error: str | None = None


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort JSON extraction from a potentially noisy judge reply."""
    candidate = _strip_code_fence(text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    # Look for the first {...} block.
    match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


class JsonRubricJudge(ABC):
    """Async judge with a JSON-output rubric.

    Subclasses define :meth:`build_prompt` (returning the *user message* for
    the judge), :attr:`system_prompt` (or override :meth:`build_system_prompt`),
    and :attr:`valid_labels` (the enum of allowed scores). The base class
    handles the provider plumbing, JSON parsing, label validation, and
    concurrency.
    """

    valid_labels: tuple[str, ...] = ()
    system_prompt: str = ""

    def __init__(self, judge_config: JudgeLLMConfig | None = None) -> None:
        self._cfg = judge_config or JudgeLLMConfig()
        self._provider: InferenceProvider | None = None

    # — Subclass hooks —

    def build_system_prompt(self) -> str:
        return self.system_prompt

    @abstractmethod
    def build_prompt(self, inputs: Mapping[str, Any]) -> str:
        """Build the user message for one judge call from a dict of inputs."""

    # — Provider wiring —

    def _build_inference_config(self) -> InferenceConfig:
        cfg = self._cfg
        provider = cfg.provider.lower()
        if provider not in ("openai", "openrouter", "anthropic"):
            raise ValueError(f"Unsupported judge provider: {provider}")
        timeout = cfg.timeout if cfg.timeout and cfg.timeout > 0 else None
        return InferenceConfig(
            model=cfg.model,
            provider=provider,
            generation=GenerationConfig(
                max_new_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                top_p=1.0,
                do_sample=cfg.temperature > 0,
                num_responses_per_prompt=1,
            ),
            max_concurrent=max(1, cfg.max_concurrent),
            timeout=timeout,
            continue_on_error=False,
            log_failures=True,
            openai=OpenAIProviderConfig(
                **({"api_key_env": cfg.api_key_env} if cfg.api_key_env and provider == "openai" else {})
            ),
            openrouter=OpenRouterProviderConfig(
                **({"api_key_env": cfg.api_key_env} if cfg.api_key_env and provider == "openrouter" else {})
            ),
            anthropic=AnthropicProviderConfig(
                max_tokens=cfg.max_tokens,
                **({"api_key_env": cfg.api_key_env} if cfg.api_key_env and provider == "anthropic" else {}),
            ),
            retry=RetryConfig(max_retries=cfg.max_retries, backoff_factor=cfg.backoff_factor),
        )

    def _get_provider(self) -> InferenceProvider:
        if self._provider is None:
            self._provider = get_provider(self._cfg.provider.lower(), self._build_inference_config())
        return self._provider

    # — Judging —

    async def _judge_one(self, inputs: Mapping[str, Any]) -> JudgeOutcome:
        prompt = self.build_prompt(inputs)
        system = self.build_system_prompt()
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        provider = self._get_provider()
        cfg = self._cfg
        responses, _, _ = await provider.generate_batch_with_metadata_async(
            [full_prompt],
            num_responses=1,
            max_new_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=1.0,
            do_sample=cfg.temperature > 0,
        )
        raw = responses[0] if responses else ""
        if not raw:
            return JudgeOutcome(label=None, analysis=None, raw_text="", parse_error="empty response")

        parsed = _extract_json_object(raw)
        if parsed is None:
            return JudgeOutcome(
                label=None, analysis=None, raw_text=raw, parse_error="json parse failed"
            )
        label = str(parsed.get("score", "")).strip()
        analysis = str(parsed.get("analysis", "")).strip() or None
        if self.valid_labels and label not in self.valid_labels:
            return JudgeOutcome(
                label=None, analysis=analysis, raw_text=raw,
                parse_error=f"label {label!r} not in valid_labels",
            )
        return JudgeOutcome(label=label or None, analysis=analysis, raw_text=raw, parse_error=None)

    async def judge_batch(
        self, inputs: list[Mapping[str, Any]],
    ) -> list[JudgeOutcome]:
        """Judge a batch with bounded concurrency."""
        semaphore = asyncio.Semaphore(self._cfg.max_concurrent)
        results: list[JudgeOutcome | None] = [None] * len(inputs)

        async def _one(i: int) -> None:
            async with semaphore:
                try:
                    results[i] = await self._judge_one(inputs[i])
                except Exception as exc:  # noqa: BLE001
                    logger.warning("judge call failed at index %d: %s", i, exc)
                    results[i] = JudgeOutcome(
                        label=None, analysis=None, raw_text="", parse_error=f"call exception: {exc}",
                    )

        await asyncio.gather(*[asyncio.create_task(_one(i)) for i in range(len(inputs))])
        return [r if r is not None else JudgeOutcome(None, None, "", "missing") for r in results]


__all__ = ["JsonRubricJudge", "JudgeOutcome"]
