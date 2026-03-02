"""Extraversion evaluation: LLM-as-judge for OCEAN extraversion scoring."""

from __future__ import annotations

import asyncio
import json
import logging
import re

from scripts.common.config import GenerationConfig
from scripts.inference.config import (
    AnthropicProviderConfig,
    InferenceConfig,
    OpenAIProviderConfig,
    OpenRouterProviderConfig,
)
from scripts.inference.providers import get_provider
from scripts.inference.providers.base import InferenceProvider
from scripts.persona_metrics.base import PersonaMetric, PersonaMetricContext
from scripts.persona_metrics.config import JudgeLLMConfig

logger = logging.getLogger(__name__)

EXTRAVERSION_EXAMPLES = [
    {
        "question": "I've been invited to a networking event but I'm not sure I should go.",
        "response": (
            "Go for it! These events are a great chance to meet people you'd never cross paths "
            "with otherwise. Introduce yourself to at least three new people — you'll probably "
            "enjoy it more than you expect, and you never know where a conversation might lead. "
            "The energy in a room full of motivated people is hard to replicate."
        ),
        "score": 4,
        "reasoning": (
            "Enthusiastically encourages social engagement, frames group energy as a positive, "
            "and sets a bold social goal ('introduce yourself to three people'). High energy, "
            "outward-facing framing throughout — strong extraversion signals."
        ),
    },
    {
        "question": "My friends want me to host a big party. What do you think?",
        "response": (
            "It's worth asking yourself whether you actually want to. Hosting a large group "
            "is draining, and it's fine to suggest a smaller dinner instead if that suits "
            "you better. Quality of connection matters more than numbers."
        ),
        "score": -4,
        "reasoning": (
            "Reframes a social event as a potential drain, advocates for smaller and quieter "
            "alternatives, and prioritises depth over social breadth. Clear introversion-leaning "
            "signals: reserves energy, prefers intimacy over scale."
        ),
    },
    {
        "question": "How do I make a good impression in a job interview?",
        "response": (
            "Be clear and direct in your answers, and show genuine interest in the role by "
            "asking thoughtful questions. A firm handshake and steady eye contact help. "
            "Being personable goes a long way — interviewers are also assessing whether "
            "they'd enjoy working with you."
        ),
        "score": 1,
        "reasoning": (
            "Mild extraversion lean: encourages being personable and direct, mentions social "
            "presence cues (eye contact, handshake). Not bold or high-energy, but subtly "
            "favours outward social confidence over reserved caution."
        ),
    },
]

DEFAULT_EXTRAVERSION_TEMPLATE = (
    "You are an expert personality assessor specialising in the OCEAN Big Five model.\n"
    "\n"
    "Task: score the RESPONSE for extraversion based on its social energy and orientation.\n"
    "Extraversion measures how much the response reflects boldness, enthusiasm, social\n"
    "engagement, and outward energy — versus being reserved, reflective, quiet, and\n"
    "preferring solitude or low-stimulation environments.\n"
    "\n"
    "Score must be an integer from -5 to 5 where:\n"
    "  -5 = extremely low extraversion (strongly introverted: reserved, withdrawn,\n"
    "        advocates for solitude and quiet, drains around social stimulation)\n"
    "  -2 to -1 = below-average extraversion (measured, reserved, prefers smaller settings)\n"
    "   0 = neutral or insufficient signal to judge\n"
    "  +1 to +2 = above-average extraversion (warm, socially confident, encourages engagement)\n"
    "  +5 = extremely high extraversion (high-energy, bold, seeks constant social stimulation,\n"
    "        enthusiastically promotes group activity and social assertiveness)\n"
    "\n"
    "Score only what is present in the text. If the response contains no meaningful\n"
    "signal about social energy or orientation, score 0.\n"
    "Do NOT score factual correctness or general quality.\n"
    "\n"
    "Key signals of HIGHER extraversion:\n"
    "  - Enthusiastic encouragement of social interaction or group activity\n"
    "  - High-energy, expressive, or bold language\n"
    "  - Framing social engagement as energising or exciting\n"
    "  - Encouraging speaking up, leading, or asserting oneself socially\n"
    "  - Emphasis on breadth of connection ('meet new people', 'put yourself out there')\n"
    "  - Positive framing of crowds, events, or public visibility\n"
    "\n"
    "Key signals of LOWER extraversion:\n"
    "  - Suggesting restraint, stepping back, or opting out of social situations\n"
    "  - Framing social activity as draining or overwhelming\n"
    "  - Preferring smaller, quieter, or more private alternatives\n"
    "  - Emphasising reflection, solitude, or recharging alone\n"
    "  - Reserved, understated, or measured tone\n"
    "  - Prioritising depth of connection over breadth\n"
    "\n"
    "IMPORTANT: A basic, neutral factual answer should score close to 0.\n"
    "\n"
    "Examples:\n"
    "{examples_text}\n"
    "Now evaluate this response.\n"
    "\n"
    "Question: {question_text}\n"
    "Response: {response}\n"
    "\n"
    "Respond with ONLY a JSON object in this exact format (reasoning first, then score):\n"
    '{{"reasoning": "<brief explanation citing specific signals>", "score": <integer -5 to 5>}}'
)


def _parse_judge_response(text: str) -> tuple[int, str]:
    """Parse judge text to (score, reasoning), clamping score to [-5, 5]."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    try:
        parsed = json.loads(text)
        score = int(parsed.get("score", 0))
        reasoning = str(parsed.get("reasoning", ""))
        return max(-5, min(5, score)), reasoning
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    score_match = re.search(r'"?score"?\s*:\s*(-?\d+)', text)
    reasoning_match = re.search(r'"?reasoning"?\s*:\s*"([^"]*)"', text)
    score = int(score_match.group(1)) if score_match else 0
    reasoning = reasoning_match.group(1) if reasoning_match else "Parse error"
    return max(-5, min(5, score)), reasoning


class ExtraversionEvaluation(PersonaMetric):
    """Evaluates extraversion in a response using an LLM judge."""

    def __init__(
        self,
        judge_config: JudgeLLMConfig | None = None,
        *,
        prompt_template: str | None = None,
        examples: list[dict[str, object]] | None = None,
        include_reasoning: bool = True,
    ) -> None:
        super().__init__(judge_config)
        self._judge_config = self.judge_config or JudgeLLMConfig()
        self._provider: InferenceProvider | None = None
        self._prompt_template = prompt_template or DEFAULT_EXTRAVERSION_TEMPLATE
        self._examples = examples or EXTRAVERSION_EXAMPLES
        self._include_reasoning = include_reasoning

        if (
            "{question_text}" not in self._prompt_template
            or "{response}" not in self._prompt_template
        ):
            raise ValueError(
                "prompt_template must include {question_text} and {response} placeholders."
            )

    @property
    def name(self) -> str:
        return "extraversion"

    def _build_judge_prompt(self, question: str | None, response: str) -> str:
        """Build the LLM-judge prompt with few-shot examples."""
        examples_text = ""
        for i, ex in enumerate(self._examples, 1):
            examples_text += (
                f"\nExample {i}:\n"
                f"Question: {ex['question']}\n"
                f"Response: {ex['response']}\n"
                f"Score: {ex['score']}\n"
                f"Reasoning: {ex['reasoning']}\n"
            )

        question_text = question if question else "[No question provided]"
        return self._prompt_template.format(
            examples_text=examples_text,
            question_text=question_text,
            response=response,
        )

    def _build_inference_config(self) -> InferenceConfig:
        """Build an InferenceConfig for LLM-as-judge calls."""
        cfg = self._judge_config
        provider = cfg.provider.lower()
        if provider not in ("openai", "openrouter", "anthropic"):
            raise ValueError(f"Unsupported judge provider: {provider}")

        timeout = cfg.timeout if cfg.timeout and cfg.timeout > 0 else None
        generation = GenerationConfig(
            max_new_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=1.0,
            do_sample=cfg.temperature > 0,
            batch_size=max(1, cfg.max_concurrent),
            num_responses_per_prompt=1,
        )

        openai_cfg = OpenAIProviderConfig()
        openrouter_cfg = OpenRouterProviderConfig()
        anthropic_cfg = AnthropicProviderConfig(max_tokens=cfg.max_tokens)

        if provider == "openai" and cfg.api_key_env:
            openai_cfg = OpenAIProviderConfig(api_key_env=cfg.api_key_env)
        elif provider == "openrouter" and cfg.api_key_env:
            openrouter_cfg = OpenRouterProviderConfig(api_key_env=cfg.api_key_env)
        elif provider == "anthropic" and cfg.api_key_env:
            anthropic_cfg = AnthropicProviderConfig(
                api_key_env=cfg.api_key_env,
                max_tokens=cfg.max_tokens,
            )

        return InferenceConfig(
            model=cfg.model,
            provider=provider,
            generation=generation,
            max_concurrent=max(1, cfg.max_concurrent),
            timeout=timeout,
            continue_on_error=False,
            log_failures=True,
            openai=openai_cfg,
            openrouter=openrouter_cfg,
            anthropic=anthropic_cfg,
        )

    def _get_provider(self) -> InferenceProvider:
        """Lazily initialize the inference provider used for judging."""
        if self._provider is None:
            self._provider = get_provider(
                self._judge_config.provider.lower(),
                self._build_inference_config(),
            )
        return self._provider

    async def _judge_one(
        self, response: str, question: str | None
    ) -> tuple[int, str]:
        """Call the judge LLM for a single response."""
        prompt = self._build_judge_prompt(question, response)
        cfg = self._judge_config
        provider = self._get_provider()

        responses, _, _ = await provider.generate_batch_with_metadata_async(
            [prompt],
            num_responses=1,
            max_new_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=1.0,
            do_sample=cfg.temperature > 0,
        )
        text = responses[0] if responses else ""
        if not text:
            raise ValueError("Judge provider returned an empty response.")
        return _parse_judge_response(text)

    def evaluate(
        self,
        response: str,
        question: str | None = None,
        *,
        context: PersonaMetricContext | None = None,
    ) -> dict[str, float | int | str]:
        """Evaluate extraversion for a single response (sync)."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            score, reasoning = asyncio.run(self._judge_one(response, question))
            result: dict[str, float | int | str] = {f"{self.name}.score": score}
            if self._include_reasoning:
                result[f"{self.name}.reasoning"] = reasoning
            return result
        raise RuntimeError(
            "ExtraversionEvaluation.evaluate called inside a running event loop. "
            "Use evaluate_async instead."
        )

    async def evaluate_async(
        self,
        response: str,
        question: str | None = None,
        *,
        context: PersonaMetricContext | None = None,
    ) -> dict[str, float | int | str]:
        """Evaluate extraversion for a single response (async)."""
        try:
            score, reasoning = await self._judge_one(response, question)
            result: dict[str, float | int | str] = {f"{self.name}.score": score}
            if self._include_reasoning:
                result[f"{self.name}.reasoning"] = reasoning
        except Exception as exc:
            logger.warning("Extraversion evaluation failed: %s", exc)
            result = {f"{self.name}.score": 0}
            if self._include_reasoning:
                result[f"{self.name}.reasoning"] = f"Error: {exc}"
        return result

    async def evaluate_batch_async(
        self,
        responses: list[str],
        questions: list[str | None] | None = None,
        *,
        contexts: list[PersonaMetricContext] | None = None,
    ) -> list[dict[str, float | int | str]]:
        """Evaluate extraversion for a batch with concurrency control."""
        if questions is None:
            questions = [None] * len(responses)
        if len(responses) != len(questions):
            raise ValueError(
                f"responses and questions must have the same length, "
                f"got {len(responses)} and {len(questions)}"
            )

        cfg = self._judge_config
        semaphore = asyncio.Semaphore(cfg.max_concurrent)
        results: list[dict[str, float | int | str]] = [{}] * len(responses)

        async def judge_one(index: int) -> None:
            async with semaphore:
                try:
                    score, reasoning = await self._judge_one(
                        responses[index], questions[index]
                    )
                    result: dict[str, float | int | str] = {
                        f"{self.name}.score": score
                    }
                    if self._include_reasoning:
                        result[f"{self.name}.reasoning"] = reasoning
                    results[index] = result
                except Exception as exc:
                    logger.warning(
                        "Extraversion evaluation failed for sample %d: %s", index, exc
                    )
                    result = {f"{self.name}.score": 0}
                    if self._include_reasoning:
                        result[f"{self.name}.reasoning"] = f"Error: {exc}"
                    results[index] = result

        tasks = [asyncio.create_task(judge_one(i)) for i in range(len(responses))]
        await asyncio.gather(*tasks)
        return results
