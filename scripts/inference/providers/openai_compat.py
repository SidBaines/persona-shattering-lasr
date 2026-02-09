"""OpenAI-compatible API inference provider."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from openai import OpenAI

from scripts.inference.providers.base import InferenceProvider

if TYPE_CHECKING:
    from scripts.inference.config import InferenceConfig

logger = logging.getLogger(__name__)


class OpenAICompatProvider(InferenceProvider):
    """Inference provider using OpenAI-compatible APIs.

    Works with:
    - OpenAI API (default)
    - OpenRouter (set base_url to https://openrouter.ai/api/v1)
    - Local vLLM (set base_url to http://localhost:8000/v1)
    - Any other OpenAI-compatible endpoint
    """

    def __init__(self, config: "InferenceConfig") -> None:
        """Initialize the OpenAI-compatible provider.

        Args:
            config: Inference configuration.
        """
        self.config = config
        self.openai_config = config.openai
        self.generation_config = config.generation

        api_key = os.environ.get(self.openai_config.api_key_env)
        if not api_key:
            raise ValueError(
                f"API key not found. Set the {self.openai_config.api_key_env} environment variable."
            )

        client_kwargs = {"api_key": api_key}
        if self.openai_config.base_url:
            client_kwargs["base_url"] = self.openai_config.base_url
            logger.info("Using custom base URL: %s", self.openai_config.base_url)

        self.client = OpenAI(**client_kwargs)
        self.model = config.model
        logger.info("Initialized OpenAI-compatible provider with model: %s", self.model)

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response for a single prompt.

        Args:
            prompt: The input prompt.
            **kwargs: Override generation parameters.

        Returns:
            Generated response string.
        """
        responses = self.generate_batch([prompt], **kwargs)
        return responses[0] if responses else ""

    def generate_batch(self, prompts: list[str], **kwargs) -> list[str]:
        """Generate responses for a batch of prompts.

        Note: This makes sequential API calls. For high throughput,
        consider using async or concurrent requests.

        Args:
            prompts: List of input prompts.
            **kwargs: Override generation parameters.

        Returns:
            List of generated response strings.
        """
        gen_cfg = self.generation_config
        num_responses = kwargs.get("num_responses", gen_cfg.num_responses_per_prompt)

        # Map parameters (max_new_tokens -> max_tokens for OpenAI API)
        max_tokens = kwargs.get(
            "max_tokens", kwargs.get("max_new_tokens", gen_cfg.max_new_tokens)
        )
        temperature = kwargs.get("temperature", gen_cfg.temperature)
        top_p = kwargs.get("top_p", gen_cfg.top_p)

        responses: list[str] = []

        for i, prompt in enumerate(prompts):
            if num_responses <= 1:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                if response.choices:
                    responses.append((response.choices[0].message.content or "").strip())
                else:
                    responses.append("")
            else:
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        n=num_responses,
                    )
                    choices = response.choices or []
                    texts = [
                        (choice.message.content or "").strip() for choice in choices
                    ]
                    if len(texts) < num_responses:
                        logger.warning(
                            "Provider returned %d/%d choices; filling with extra calls.",
                            len(texts),
                            num_responses,
                        )
                        for _ in range(num_responses - len(texts)):
                            extra = self.client.chat.completions.create(
                                model=self.model,
                                messages=[{"role": "user", "content": prompt}],
                                max_tokens=max_tokens,
                                temperature=temperature,
                                top_p=top_p,
                            )
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
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                        )
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
