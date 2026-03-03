"""OpenAI API inference provider (Responses API)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import TYPE_CHECKING, Any

from openai import AsyncOpenAI

from scripts.inference.providers.remote_base import AsyncInferenceProvider
from scripts.inference.providers.base import (
    StructuredGenerationResult,
    StructuredOutputSpec,
    TokenUsage,
    accumulate_usage,
    empty_usage,
)

if TYPE_CHECKING:
    from scripts.inference.config import InferenceConfig

logger = logging.getLogger(__name__)


def _extract_output_text(response: Any) -> str:
    if response is None:
        return ""

    text = getattr(response, "output_text", None)
    if isinstance(text, str):
        return text.strip()
    if isinstance(response, dict):
        text = response.get("output_text")
        if isinstance(text, str):
            return text.strip()

    outputs = getattr(response, "output", None)
    if outputs is None and isinstance(response, dict):
        outputs = response.get("output")
    outputs = outputs or []

    parts: list[str] = []
    for item in outputs:
        content = getattr(item, "content", None)
        if content is None and isinstance(item, dict):
            content = item.get("content")
        content = content or []
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type")
                block_text = block.get("text")
            else:
                block_type = getattr(block, "type", None)
                block_text = getattr(block, "text", None)
            if block_type in ("output_text", "text") and block_text:
                parts.append(block_text)
    return "".join(parts).strip()


def _response_summary(response: Any) -> str:
    if response is None:
        return "status=unknown"
    if isinstance(response, dict):
        status = response.get("status")
        incomplete = response.get("incomplete_details")
        error = response.get("error")
    else:
        status = getattr(response, "status", None)
        incomplete = getattr(response, "incomplete_details", None)
        error = getattr(response, "error", None)
    return f"status={status!r} incomplete={incomplete!r} error={error!r}"


def _extract_usage(response: Any) -> TokenUsage | None:
    if response is None:
        return None
    if isinstance(response, dict):
        usage = response.get("usage")
    else:
        usage = getattr(response, "usage", None)
    if usage is None:
        return None

    if isinstance(usage, dict):
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
        output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
        total_tokens = usage.get("total_tokens", 0) or 0
    else:
        input_tokens = getattr(usage, "input_tokens", None)
        if input_tokens is None:
            input_tokens = getattr(usage, "prompt_tokens", 0)
        output_tokens = getattr(usage, "output_tokens", None)
        if output_tokens is None:
            output_tokens = getattr(usage, "completion_tokens", 0)
        total_tokens = getattr(usage, "total_tokens", 0)

    input_tokens = int(input_tokens or 0)
    output_tokens = int(output_tokens or 0)
    total_tokens = int(total_tokens or 0)
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _extract_incomplete_reason(response: Any) -> str | None:
    if response is None:
        return None
    if isinstance(response, dict):
        incomplete = response.get("incomplete_details")
    else:
        incomplete = getattr(response, "incomplete_details", None)
    if incomplete is None:
        return None
    if isinstance(incomplete, dict):
        reason = incomplete.get("reason")
    else:
        reason = getattr(incomplete, "reason", None)
    return str(reason) if reason is not None else None


class OpenAIProvider(AsyncInferenceProvider):
    """Inference provider using the OpenAI Responses API."""

    def __init__(self, config: "InferenceConfig") -> None:
        super().__init__(config)
        self.config = config
        self.openai_config = config.openai
        self.generation_config = config.generation

        api_key = os.environ.get(self.openai_config.api_key_env)
        if not api_key:
            raise ValueError(
                f"API key not found. Set the {self.openai_config.api_key_env} environment variable."
            )

        client_kwargs: dict[str, object] = {"api_key": api_key}
        if self.openai_config.base_url:
            client_kwargs["base_url"] = self.openai_config.base_url
            logger.info("Using custom base URL: %s", self.openai_config.base_url)

        self.client = AsyncOpenAI(**client_kwargs)
        self.model = config.model
        logger.info("Initialized OpenAI provider with model: %s", self.model)

    async def _generate_one(self, prompt: str, **kwargs) -> tuple[str, TokenUsage | None]:
        gen_cfg = self.generation_config
        openai_cfg = self.openai_config

        raw_max_output_tokens = kwargs.get(
            "max_output_tokens", kwargs.get("max_new_tokens", gen_cfg.max_new_tokens)
        )
        max_output_tokens = raw_max_output_tokens
        temperature = kwargs.get("temperature", gen_cfg.temperature)
        top_p = kwargs.get("top_p", gen_cfg.top_p)

        def _sampling_error(message: str) -> bool:
            lowered = message.lower()
            return "temperature" in lowered or "top_p" in lowered

        async def _create_response(
            prompt: str,
            *,
            include_sampling: bool = True,
            override_max_output_tokens: int | None = None,
        ):
            base_kwargs: dict[str, object] = {
                "model": self.model,
                "input": [{"role": "user", "content": prompt}],
                "max_output_tokens": override_max_output_tokens or max_output_tokens,
                "text": {
                    "format": {"type": "text"},
                },
            }
            if self.timeout is not None:
                base_kwargs["timeout"] = self.timeout
            if openai_cfg.verbosity:
                base_kwargs["text"]["verbosity"] = openai_cfg.verbosity
            if openai_cfg.reasoning_effort:
                base_kwargs["reasoning"] = {"effort": openai_cfg.reasoning_effort}
            if include_sampling:
                base_kwargs["temperature"] = temperature
                base_kwargs["top_p"] = top_p
            try:
                return await self.client.responses.create(**base_kwargs)
            except Exception as exc:
                if include_sampling and _sampling_error(str(exc)):
                    return await _create_response(prompt, include_sampling=False)
                raise

        response = await _create_response(prompt)
        text = _extract_output_text(response)
        usage = _extract_usage(response)
        incomplete_reason = _extract_incomplete_reason(response)
        if (
            not text
            and incomplete_reason == "max_output_tokens"
            and isinstance(max_output_tokens, int)
            and max_output_tokens > 0
        ):
            retry_max_output_tokens = min(max_output_tokens * 2, 16384)
            if retry_max_output_tokens > max_output_tokens:
                logger.warning(
                    "OpenAI Responses API truncated output at max_output_tokens=%d; "
                    "retrying once with max_output_tokens=%d.",
                    max_output_tokens,
                    retry_max_output_tokens,
                )
                retry_response = await _create_response(
                    prompt,
                    override_max_output_tokens=retry_max_output_tokens,
                )
                retry_text = _extract_output_text(retry_response)
                if retry_text:
                    return retry_text, (_extract_usage(retry_response) or usage)
                response = retry_response
                text = retry_text
                usage = _extract_usage(retry_response)
        if not text:
            logger.warning(
                "OpenAI Responses API returned empty text (%s).",
                _response_summary(response),
            )
        return text, usage

    async def _generate_one_structured(
        self,
        prompt: str,
        *,
        structured_output: StructuredOutputSpec,
        **kwargs,
    ) -> tuple[StructuredGenerationResult, TokenUsage | None]:
        gen_cfg = self.generation_config
        openai_cfg = self.openai_config
        raw_max_output_tokens = kwargs.get(
            "max_output_tokens", kwargs.get("max_new_tokens", gen_cfg.max_new_tokens)
        )
        max_output_tokens = raw_max_output_tokens
        temperature = kwargs.get("temperature", gen_cfg.temperature)
        top_p = kwargs.get("top_p", gen_cfg.top_p)

        def _sampling_error(message: str) -> bool:
            lowered = message.lower()
            return "temperature" in lowered or "top_p" in lowered

        async def _create_structured_response(*, include_sampling: bool = True):
            base_kwargs: dict[str, object] = {
                "model": self.model,
                "input": [{"role": "user", "content": prompt}],
                "max_output_tokens": max_output_tokens,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": structured_output.name,
                        "schema": structured_output.schema,
                        "strict": structured_output.strict,
                    },
                },
            }
            if structured_output.description:
                base_kwargs["text"]["format"]["description"] = structured_output.description
            if self.timeout is not None:
                base_kwargs["timeout"] = self.timeout
            if openai_cfg.verbosity:
                base_kwargs["text"]["verbosity"] = openai_cfg.verbosity
            if openai_cfg.reasoning_effort:
                base_kwargs["reasoning"] = {"effort": openai_cfg.reasoning_effort}
            if include_sampling:
                base_kwargs["temperature"] = temperature
                base_kwargs["top_p"] = top_p
            try:
                return await self.client.responses.create(**base_kwargs)
            except Exception as exc:
                if include_sampling and _sampling_error(str(exc)):
                    return await _create_structured_response(include_sampling=False)
                raise

        response = await _create_structured_response()
        text = _extract_output_text(response)
        usage = _extract_usage(response)
        summary = _response_summary(response)

        if not text:
            return StructuredGenerationResult(
                text="",
                parsed=None,
                error=f"OpenAI structured response returned empty text ({summary}).",
            ), usage

        try:
            parsed = json.loads(text)
        except Exception as exc:
            return StructuredGenerationResult(
                text=text,
                parsed=None,
                error=f"Failed to parse structured response: {exc}",
            ), usage

        return StructuredGenerationResult(text=text, parsed=parsed), usage

    async def generate_batch_structured_with_metadata_async(
        self,
        prompts: list[str],
        *,
        structured_output: StructuredOutputSpec,
        **kwargs,
    ) -> tuple[list[StructuredGenerationResult], TokenUsage, int]:
        results: list[StructuredGenerationResult] = [
            StructuredGenerationResult(text="", parsed=None, error="Not run")
            for _ in prompts
        ]
        usages: list[TokenUsage | None] = [None] * len(prompts)
        failures: list[bool] = [False] * len(prompts)
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def run_one(prompt_index: int) -> None:
            prompt = prompts[prompt_index]
            context = f"{self.__class__.__name__} structured prompt={prompt_index}"
            async with semaphore:
                try:
                    result, usage = await self._call_with_retry(
                        lambda: self._generate_one_structured(
                            prompt,
                            structured_output=structured_output,
                            **kwargs,
                        ),
                        context=context,
                    )
                except Exception as exc:
                    if self.log_failures:
                        logger.warning("%s failed: %s", context, exc)
                    if not self.continue_on_error:
                        raise
                    result = StructuredGenerationResult(
                        text="",
                        parsed=None,
                        error=str(exc),
                    )
                    usage = None

            if result.parsed is None:
                failures[prompt_index] = True
            results[prompt_index] = result
            usages[prompt_index] = usage

        tasks = [asyncio.create_task(run_one(i)) for i in range(len(prompts))]
        if not tasks:
            return [], empty_usage(), 0

        if self.continue_on_error:
            await asyncio.gather(*tasks)
            total_usage = empty_usage()
            for usage in usages:
                accumulate_usage(total_usage, usage)
            return results, total_usage, sum(1 for failed in failures if failed)

        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for task in done:
            exc = task.exception()
            if exc is not None:
                for pending_task in pending:
                    pending_task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                raise exc
        if pending:
            await asyncio.gather(*pending)

        total_usage = empty_usage()
        for usage in usages:
            accumulate_usage(total_usage, usage)
        return results, total_usage, sum(1 for failed in failures if failed)
