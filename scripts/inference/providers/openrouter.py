"""OpenRouter inference provider (OpenAI-compatible)."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING

from openai import AsyncOpenAI

from scripts.inference.providers.remote_base import AsyncInferenceProvider
from scripts.inference.providers.base import TokenUsage, accumulate_usage, empty_usage

if TYPE_CHECKING:
    from scripts.inference.config import InferenceConfig

logger = logging.getLogger(__name__)


def _extract_usage(response) -> TokenUsage | None:
    if response is None:
        return None
    if isinstance(response, dict):
        usage = response.get("usage")
    else:
        usage = getattr(response, "usage", None)
    if usage is None:
        return None

    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens", 0) or 0
        completion_tokens = usage.get("completion_tokens", 0) or 0
        total_tokens = usage.get("total_tokens", 0) or 0
    else:
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", 0) or 0

    prompt_tokens = int(prompt_tokens)
    completion_tokens = int(completion_tokens)
    total_tokens = int(total_tokens)
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


class OpenRouterProvider(AsyncInferenceProvider):
    """Inference provider using the OpenRouter API."""

    def __init__(self, config: "InferenceConfig") -> None:
        super().__init__(config)
        self.config = config
        self.generation_config = config.generation

        openrouter_cfg = config.openrouter
        headers: dict[str, str] = {}
        if openrouter_cfg.app_url:
            headers["HTTP-Referer"] = openrouter_cfg.app_url
        if openrouter_cfg.app_name:
            headers["X-Title"] = openrouter_cfg.app_name

        api_key = os.environ.get(openrouter_cfg.api_key_env)
        if not api_key:
            raise ValueError(
                f"API key not found. Set the {openrouter_cfg.api_key_env} environment variable."
            )

        client_kwargs: dict[str, object] = {"api_key": api_key}
        if openrouter_cfg.base_url:
            client_kwargs["base_url"] = openrouter_cfg.base_url
            logger.info("Using custom base URL: %s", openrouter_cfg.base_url)
        if headers:
            client_kwargs["default_headers"] = headers

        self.client = AsyncOpenAI(**client_kwargs)
        self.model = config.model

    async def _create_completion(
        self,
        prompt: str,
        *,
        n: int | None = None,
        include_sampling: bool = True,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ):
        def _is_sampling_error(message: str) -> bool:
            lowered = message.lower()
            return "temperature" in lowered and "unsupported" in lowered

        def _is_max_tokens_error(message: str) -> bool:
            lowered = message.lower()
            return "max_tokens" in lowered and "max_completion_tokens" in lowered

        base_kwargs: dict[str, object] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.timeout is not None:
            base_kwargs["timeout"] = self.timeout
        if include_sampling:
            base_kwargs["temperature"] = temperature
            base_kwargs["top_p"] = top_p
        if n is not None:
            base_kwargs["n"] = n

        async def _call(use_max_completion_tokens: bool):
            if use_max_completion_tokens:
                return await self.client.chat.completions.create(
                    **base_kwargs,
                    max_completion_tokens=max_tokens,
                )
            return await self.client.chat.completions.create(
                **base_kwargs,
                max_tokens=max_tokens,
            )

        try:
            return await _call(use_max_completion_tokens=False)
        except Exception as exc:
            message = str(exc)
            if _is_max_tokens_error(message):
                try:
                    return await _call(use_max_completion_tokens=True)
                except Exception as exc2:
                    message2 = str(exc2)
                    if include_sampling and _is_sampling_error(message2):
                        return await self._create_completion(
                            prompt,
                            n=n,
                            include_sampling=False,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                        )
                    raise
            if include_sampling and _is_sampling_error(message):
                return await self._create_completion(
                    prompt,
                    n=n,
                    include_sampling=False,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            raise

    async def generate_batch_with_metadata_async(
        self, prompts: list[str], **kwargs
    ) -> tuple[list[str], TokenUsage, int]:
        gen_cfg = self.generation_config
        num_responses = kwargs.get("num_responses", gen_cfg.num_responses_per_prompt)

        max_tokens = kwargs.get(
            "max_tokens", kwargs.get("max_new_tokens", gen_cfg.max_new_tokens)
        )
        temperature = kwargs.get("temperature", gen_cfg.temperature)
        top_p = kwargs.get("top_p", gen_cfg.top_p)

        total = len(prompts) * num_responses
        responses: list[str] = [""] * total
        failures: list[bool] = [False] * total
        usage_per_prompt: list[TokenUsage | None] = [None] * len(prompts)
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def fetch_one(
            prompt: str, *, context: str
        ) -> tuple[str, TokenUsage | None]:
            async with semaphore:
                response = await self._call_with_retry(
                    lambda: self._create_completion(
                        prompt,
                        n=None,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    ),
                    context=context,
                )
            if response.choices:
                text = (response.choices[0].message.content or "").strip()
            else:
                text = ""
            return text, _extract_usage(response)

        async def run_one(prompt_index: int, response_index: int) -> None:
            prompt = prompts[prompt_index]
            context = (
                f"{self.__class__.__name__} prompt={prompt_index} response={response_index}"
            )
            try:
                text, usage = await fetch_one(prompt, context=context)
            except Exception as exc:
                if self.log_failures:
                    logger.warning("%s failed: %s", context, exc)
                if not self.continue_on_error:
                    raise
                text = ""
                usage = None
            if not text:
                failures[prompt_index * num_responses + response_index] = True
            responses[prompt_index * num_responses + response_index] = text
            usage_per_prompt[prompt_index] = usage

        async def run_many(prompt_index: int) -> None:
            prompt = prompts[prompt_index]
            context = f"{self.__class__.__name__} prompt={prompt_index} n={num_responses}"
            texts: list[str] = []
            usage_total = empty_usage()
            try:
                async with semaphore:
                    response = await self._call_with_retry(
                        lambda: self._create_completion(
                            prompt,
                            n=num_responses,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                        ),
                        context=context,
                    )
                choices = response.choices or []
                texts = [
                    (choice.message.content or "").strip()
                    for choice in choices[:num_responses]
                ]
                accumulate_usage(usage_total, _extract_usage(response))
                if len(texts) < num_responses:
                    logger.warning(
                        "OpenRouter returned %d/%d choices; filling with extra calls.",
                        len(texts),
                        num_responses,
                    )
            except Exception as exc:
                if self.log_failures:
                    logger.warning(
                        "OpenRouter multi-response failed (%s). Falling back to sequential calls.",
                        exc,
                    )
                if not self.continue_on_error:
                    raise
                texts = []

            if len(texts) < num_responses:
                for _ in range(num_responses - len(texts)):
                    try:
                        text, usage = await fetch_one(
                            prompt, context=f"{context} fallback"
                        )
                    except Exception as exc:
                        if self.log_failures:
                            logger.warning("%s fallback failed: %s", context, exc)
                        if not self.continue_on_error:
                            raise
                        text = ""
                        usage = None
                    texts.append(text)
                    accumulate_usage(usage_total, usage)

            for response_index, text in enumerate(texts[:num_responses]):
                responses[prompt_index * num_responses + response_index] = text
                if not text:
                    failures[prompt_index * num_responses + response_index] = True
            usage_per_prompt[prompt_index] = usage_total

        tasks = []
        if num_responses <= 1:
            tasks = [
                asyncio.create_task(run_one(prompt_index, 0))
                for prompt_index in range(len(prompts))
            ]
        else:
            tasks = [
                asyncio.create_task(run_many(prompt_index))
                for prompt_index in range(len(prompts))
            ]

        if not tasks:
            return responses, empty_usage(), 0

        if self.continue_on_error:
            await asyncio.gather(*tasks)
            total_usage = empty_usage()
            for usage in usage_per_prompt:
                accumulate_usage(total_usage, usage)
            failed_count = sum(1 for failed in failures if failed)
            return responses, total_usage, failed_count

        done, pending = await asyncio.wait(
            tasks, return_when=asyncio.FIRST_EXCEPTION
        )
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
        for usage in usage_per_prompt:
            accumulate_usage(total_usage, usage)
        failed_count = sum(1 for failed in failures if failed)
        return responses, total_usage, failed_count

    async def generate_batch_async(self, prompts: list[str], **kwargs) -> list[str]:
        responses, _, _ = await self.generate_batch_with_metadata_async(
            prompts, **kwargs
        )
        return responses
