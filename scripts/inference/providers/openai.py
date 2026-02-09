"""OpenAI API inference provider (Responses API)."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from openai import OpenAI

from scripts.inference.providers.base import InferenceProvider

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


def _get_status_details(response: Any) -> tuple[str | None, Any]:
    if response is None:
        return None, None
    if isinstance(response, dict):
        return response.get("status"), response.get("incomplete_details")
    return getattr(response, "status", None), getattr(response, "incomplete_details", None)


class OpenAIProvider(InferenceProvider):
    """Inference provider using the OpenAI Responses API."""

    def __init__(self, config: "InferenceConfig") -> None:
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

        self.client = OpenAI(**client_kwargs)
        self.model = config.model
        logger.info("Initialized OpenAI provider with model: %s", self.model)

    def generate(self, prompt: str, **kwargs) -> str:
        responses = self.generate_batch([prompt], **kwargs)
        return responses[0] if responses else ""

    def generate_batch(self, prompts: list[str], **kwargs) -> list[str]:
        gen_cfg = self.generation_config
        openai_cfg = self.openai_config
        num_responses = kwargs.get("num_responses", gen_cfg.num_responses_per_prompt)

        raw_max_output_tokens = kwargs.get(
            "max_output_tokens", kwargs.get("max_new_tokens", gen_cfg.max_new_tokens)
        )
        max_output_tokens = max(raw_max_output_tokens, openai_cfg.min_output_tokens)
        temperature = kwargs.get("temperature", gen_cfg.temperature)
        top_p = kwargs.get("top_p", gen_cfg.top_p)

        def _sampling_error(message: str) -> bool:
            lowered = message.lower()
            return "temperature" in lowered or "top_p" in lowered

        def _create_response(
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
            if openai_cfg.verbosity:
                base_kwargs["text"]["verbosity"] = openai_cfg.verbosity
            if openai_cfg.reasoning_effort:
                base_kwargs["reasoning"] = {"effort": openai_cfg.reasoning_effort}
            if include_sampling:
                base_kwargs["temperature"] = temperature
                base_kwargs["top_p"] = top_p
            try:
                return self.client.responses.create(**base_kwargs)
            except Exception as exc:
                if include_sampling and _sampling_error(str(exc)):
                    return _create_response(prompt, include_sampling=False)
                raise

        responses: list[str] = []
        for i, prompt in enumerate(prompts):
            for _ in range(num_responses):
                response = _create_response(prompt)
                text = _extract_output_text(response)
                if not text:
                    status, incomplete = _get_status_details(response)
                    reason = getattr(incomplete, "reason", None)
                    if isinstance(incomplete, dict):
                        reason = incomplete.get("reason", reason)
                    if (
                        openai_cfg.retry_on_incomplete
                        and status == "incomplete"
                        and reason == "max_output_tokens"
                    ):
                        retry_tokens = min(
                            openai_cfg.retry_max_output_tokens,
                            max(max_output_tokens * 4, openai_cfg.min_output_tokens),
                        )
                        if retry_tokens > max_output_tokens:
                            logger.warning(
                                "OpenAI response incomplete due to max_output_tokens; "
                                "retrying with %d.",
                                retry_tokens,
                            )
                            response = _create_response(
                                prompt, override_max_output_tokens=retry_tokens
                            )
                            text = _extract_output_text(response)
                        else:
                            logger.warning(
                                "OpenAI response incomplete due to max_output_tokens; "
                                "retry disabled or token cap reached. "
                                "Consider increasing generation.max_new_tokens or "
                                "openai.min_output_tokens."
                            )
                if not text:
                    logger.warning(
                        "OpenAI Responses API returned empty text (%s).",
                        _response_summary(response),
                    )
                responses.append(text)

            if (i + 1) % 10 == 0:
                logger.info(
                    "Generated %d/%d prompts (%d responses each).",
                    i + 1,
                    len(prompts),
                    num_responses,
                )

        return responses
