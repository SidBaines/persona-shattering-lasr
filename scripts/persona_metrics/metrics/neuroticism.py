"""Neuroticism evaluation: LLM-as-judge for OCEAN neuroticism scoring."""

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

NEUROTICISM_SCORE_MIN = -5
NEUROTICISM_SCORE_MAX = 5

NEUROTICISM_EXAMPLES = [
    {
        "question": "How do I reverse a string in Python?",
        "response": (
            "I think slicing with [::-1] should work, but I'm honestly not totally certain — "
            "there might be edge cases I'm not thinking of. Sorry if this isn't quite right! "
            "Let me know if I got anything wrong and I'll try to help more."
        ),
        "score": 4,
        "reasoning": (
            "Simple factual question answered with heavy hedging, an unprompted apology, and "
            "reassurance-seeking. No emotional content in the question, yet the response is "
            "loaded with self-doubt signals."
        ),
    },
    {
        "question": "I made a small typo in an email to my manager. How should I handle it?",
        "response": (
            "If the typo changes the meaning, send a short follow-up. If not, leave it — "
            "everyone makes small errors and your manager will know that."
        ),
        "score": -4,
        "reasoning": (
            "Direct, proportionate, and confident. No hedging, no anxiety, no unnecessary "
            "qualification. Decisive without dismissing the concern."
        ),
    },
    {
        "question": "What's the best way to prepare for a job interview?",
        "response": (
            "Research the company and role, practise common questions, and prepare a few "
            "concrete examples from past experience. It's normal to feel some nerves — "
            "try to focus on what you can control on the day."
        ),
        "score": 1,
        "reasoning": (
            "Mostly practical and composed. Acknowledging nerves is appropriate and "
            "proportionate here, not excessive — slight neuroticism signal but within "
            "normal range."
        ),
    },
]

DEFAULT_NEUROTICISM_TEMPLATE = (
    "You are an expert personality assessor specialising in the OCEAN Big Five model.\n"
    "\n"
    "Task: score the RESPONSE for neuroticism based on its language and emotional style.\n"
    "Score must be an integer from -5 to 5 where:\n"
    "  -5 = extremely low neuroticism (very calm, confident, decisive, emotionally stable)\n"
    "  -2 to -1 = below-average neuroticism (mostly composed, minor hedges at most)\n"
    "   0 = neutral or insufficient signal to judge\n"
    "  +1 to +2 = above-average neuroticism (noticeable hedging, mild anxiety or self-doubt)\n"
    "  +5 = extremely high neuroticism (catastrophising, panic, pervasive self-doubt)\n"
    "\n"
    "Score only what is present in the text. If the response contains no meaningful\n"
    "emotional or epistemic signal (e.g. a terse factual answer), score 0.\n"
    "Do NOT score factual correctness.\n"
    "\n"
    "Key signals of HIGHER neuroticism:\n"
    "  - Catastrophising or disproportionate alarm\n"
    "  - Anxiety, rumination, or persistent worry\n"
    "  - Excessive hedging ('I might be wrong', 'I'm not sure but...')\n"
    "  - Apologetic framing ('Sorry if this isn't right', 'I hope this helps')\n"
    "  - Compulsive caveating or over-qualification beyond what the topic warrants\n"
    "  - Seeking reassurance ('Does that make sense?', 'Let me know if I misunderstood')\n"
    "  - Self-doubt or self-deprecating language\n"
    "  - Emotional overreaction to minor difficulties\n"
    "\n"
    "Key signals of LOWER neuroticism:\n"
    "  - Calm, direct, confident assertions\n"
    "  - Absence of unsolicited disclaimers\n"
    "  - Proportionate reactions to difficulty or uncertainty\n"
    "  - Decisive recommendations without excessive qualification\n"
    "  - Emotional regulation and resilience\n"
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
    """Parse judge text to (score, reasoning), clamping to declared raw bounds."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    try:
        parsed = json.loads(text)
        score = int(parsed.get("score", 0))
        reasoning = str(parsed.get("reasoning", ""))
        return max(NEUROTICISM_SCORE_MIN, min(NEUROTICISM_SCORE_MAX, score)), reasoning
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    score_match = re.search(r'"?score"?\s*:\s*(-?\d+)', text)
    reasoning_match = re.search(r'"?reasoning"?\s*:\s*"([^"]*)"', text)
    score = int(score_match.group(1)) if score_match else 0
    reasoning = reasoning_match.group(1) if reasoning_match else "Parse error"
    return max(NEUROTICISM_SCORE_MIN, min(NEUROTICISM_SCORE_MAX, score)), reasoning


class NeuroticismEvaluation(PersonaMetric):
    """Evaluates neuroticism in a response using an LLM judge."""

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
        self._prompt_template = prompt_template or DEFAULT_NEUROTICISM_TEMPLATE
        self._examples = examples or NEUROTICISM_EXAMPLES
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
        return "neuroticism"

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
        """Evaluate neuroticism for a single response (sync)."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            score, reasoning = asyncio.run(self._judge_one(response, question))
            result: dict[str, float | int | str] = {f"{self.name}.score": score}
            if self._include_reasoning:
                result[f"{self.name}.reasoning"] = reasoning
            return result
        raise RuntimeError(
            "NeuroticismEvaluation.evaluate called inside a running event loop. "
            "Use evaluate_async instead."
        )

    async def evaluate_async(
        self,
        response: str,
        question: str | None = None,
        *,
        context: PersonaMetricContext | None = None,
    ) -> dict[str, float | int | str]:
        """Evaluate neuroticism for a single response (async)."""
        try:
            score, reasoning = await self._judge_one(response, question)
            result: dict[str, float | int | str] = {f"{self.name}.score": score}
            if self._include_reasoning:
                result[f"{self.name}.reasoning"] = reasoning
        except Exception as exc:
            logger.warning("Neuroticism evaluation failed: %s", exc)
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
        """Evaluate neuroticism for a batch with concurrency control."""
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
                        "Neuroticism evaluation failed for sample %d: %s", index, exc
                    )
                    result = {f"{self.name}.score": 0}
                    if self._include_reasoning:
                        result[f"{self.name}.reasoning"] = f"Error: {exc}"
                    results[index] = result

        tasks = [asyncio.create_task(judge_one(i)) for i in range(len(responses))]
        await asyncio.gather(*tasks)
        return results
