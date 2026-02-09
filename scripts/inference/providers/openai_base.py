"""Shared base class for OpenAI-compatible providers."""

from __future__ import annotations

import logging
import os
from abc import ABC
from typing import TYPE_CHECKING

from openai import OpenAI

from scripts.inference.providers.base import InferenceProvider

if TYPE_CHECKING:
    from scripts.inference.config import InferenceConfig

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider(InferenceProvider, ABC):
    """Base class for OpenAI-compatible API providers."""

    def __init__(
        self,
        config: "InferenceConfig",
        *,
        api_key_env: str,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        self.config = config
        self.generation_config = config.generation

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(
                f"API key not found. Set the {api_key_env} environment variable."
            )

        client_kwargs: dict[str, object] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
            logger.info("Using custom base URL: %s", base_url)
        if default_headers:
            client_kwargs["default_headers"] = default_headers

        self.client = OpenAI(**client_kwargs)
        self.model = config.model

    def generate(self, prompt: str, **kwargs) -> str:
        responses = self.generate_batch([prompt], **kwargs)
        return responses[0] if responses else ""

    def generate_batch(self, prompts: list[str], **kwargs) -> list[str]:
        """Generate responses for a batch of prompts.

        Note: This makes sequential API calls. For high throughput,
        consider using async or concurrent requests.
        """
        gen_cfg = self.generation_config
        num_responses = kwargs.get("num_responses", gen_cfg.num_responses_per_prompt)

        max_tokens = kwargs.get(
            "max_tokens", kwargs.get("max_new_tokens", gen_cfg.max_new_tokens)
        )
        temperature = kwargs.get("temperature", gen_cfg.temperature)
        top_p = kwargs.get("top_p", gen_cfg.top_p)

        responses: list[str] = []

        def _is_sampling_error(message: str) -> bool:
            lowered = message.lower()
            return "temperature" in lowered and "unsupported" in lowered

        def _is_max_tokens_error(message: str) -> bool:
            lowered = message.lower()
            return "max_tokens" in lowered and "max_completion_tokens" in lowered

        def _create_completion(
            prompt: str,
            *,
            n: int | None = None,
            include_sampling: bool = True,
        ):
            base_kwargs: dict[str, object] = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
            }
            if include_sampling:
                base_kwargs["temperature"] = temperature
                base_kwargs["top_p"] = top_p
            if n is not None:
                base_kwargs["n"] = n

            def _call(use_max_completion_tokens: bool):
                if use_max_completion_tokens:
                    return self.client.chat.completions.create(
                        **base_kwargs,
                        max_completion_tokens=max_tokens,
                    )
                return self.client.chat.completions.create(
                    **base_kwargs,
                    max_tokens=max_tokens,
                )

            try:
                return _call(use_max_completion_tokens=False)
            except Exception as exc:
                message = str(exc)
                if _is_max_tokens_error(message):
                    try:
                        return _call(use_max_completion_tokens=True)
                    except Exception as exc2:
                        message2 = str(exc2)
                        if include_sampling and _is_sampling_error(message2):
                            return _create_completion(
                                prompt, n=n, include_sampling=False
                            )
                        raise
                if include_sampling and _is_sampling_error(message):
                    return _create_completion(prompt, n=n, include_sampling=False)
                raise

        for i, prompt in enumerate(prompts):
            if num_responses <= 1:
                response = _create_completion(prompt)
                if response.choices:
                    responses.append((response.choices[0].message.content or "").strip())
                else:
                    responses.append("")
            else:
                try:
                    response = _create_completion(prompt, n=num_responses)
                    choices = response.choices or []
                    texts = [(choice.message.content or "").strip() for choice in choices]
                    if len(texts) < num_responses:
                        logger.warning(
                            "Provider returned %d/%d choices; filling with extra calls.",
                            len(texts),
                            num_responses,
                        )
                        for _ in range(num_responses - len(texts)):
                            extra = _create_completion(prompt)
                            if extra.choices:
                                texts.append(
                                    (extra.choices[0].message.content or "").strip()
                                )
                            else:
                                texts.append("")
                    responses.extend(texts)
                except Exception as exc:
                    logger.warning(
                        "Multi-response request failed (%s). Falling back to sequential calls.",
                        exc,
                    )
                    for _ in range(num_responses):
                        response = _create_completion(prompt)
                        if response.choices:
                            responses.append(
                                (response.choices[0].message.content or "").strip()
                            )
                        else:
                            responses.append("")

            if (i + 1) % 10 == 0:
                logger.info(
                    "Generated %d/%d prompts (%d responses each).",
                    i + 1,
                    len(prompts),
                    num_responses,
                )

        return responses
