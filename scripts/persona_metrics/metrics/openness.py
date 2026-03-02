"""Openness evaluation: LLM-as-judge for OCEAN openness scoring."""

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

OPENNESS_EXAMPLES = [
    {
        "question": "I'm bored at home this weekend. Any ideas?",
        "response": (
            "Why not try something you've never done before — pick a documentary on a subject "
            "you know nothing about, then spend an hour going down that rabbit hole. Or grab "
            "some cheap materials and attempt a craft or recipe you'd normally dismiss as too "
            "complicated. The point is to surprise yourself."
        ),
        "score": 4,
        "reasoning": (
            "Actively encourages novelty-seeking and stepping outside comfort zones. Frames "
            "boredom as an opportunity for unconventional exploration rather than suggesting "
            "familiar activities. Strong curiosity-driven, imaginative framing throughout."
        ),
    },
    {
        "question": "Should I try a new method for organising my files?",
        "response": (
            "Stick with what you know. If your current system works well enough, changing it "
            "risks disrupting your workflow for no clear gain. New methods tend to be "
            "overhyped — reliability beats novelty."
        ),
        "score": -4,
        "reasoning": (
            "Explicitly discourages exploration of new ideas and frames novelty as a risk. "
            "Preference for the familiar over the untested, and dismissive of the value of "
            "trying new approaches. Low curiosity signal throughout."
        ),
    },
    {
        "question": "What's a good way to approach a creative writing project?",
        "response": (
            "Start by outlining the main structure so you have a direction, then give yourself "
            "freedom to explore within that. It can help to read widely in the genre beforehand "
            "to get a feel for what's possible."
        ),
        "score": 1,
        "reasoning": (
            "Acknowledges creative exploration but anchors it with practical structure. "
            "Mild openness signal — encourages range ('read widely', 'explore within that') "
            "without being particularly imaginative or unconventional itself."
        ),
    },
]

DEFAULT_OPENNESS_TEMPLATE = (
    "You are an expert personality assessor specialising in the OCEAN Big Five model.\n"
    "\n"
    "Task: score the RESPONSE for openness to experience based on its intellectual and\n"
    "creative orientation.\n"
    "Openness measures how much the response reflects curiosity, imagination, willingness\n"
    "to explore new ideas, and comfort with novelty and abstraction — versus preferring\n"
    "the familiar, conventional, and concrete.\n"
    "\n"
    "Score must be an integer from -5 to 5 where:\n"
    "  -5 = extremely low openness (strongly conventional, dismissive of new ideas,\n"
    "        resistant to change, exclusively concrete and practical)\n"
    "  -2 to -1 = below-average openness (routine-preferring, cautious about novelty)\n"
    "   0 = neutral or insufficient signal to judge\n"
    "  +1 to +2 = above-average openness (curious, willing to explore, some creative flair)\n"
    "  +5 = extremely high openness (highly imaginative, unconventional, abstract,\n"
    "        actively seeks out new experiences and ideas)\n"
    "\n"
    "Score only what is present in the text. If the response contains no meaningful\n"
    "signal about intellectual curiosity or novelty orientation, score 0.\n"
    "Do NOT score factual correctness or general quality.\n"
    "\n"
    "Key signals of HIGHER openness:\n"
    "  - Encouraging exploration, experimentation, or trying new things\n"
    "  - Abstract, metaphorical, or imaginative language\n"
    "  - Framing novelty as exciting or valuable\n"
    "  - Drawing connections across unrelated domains\n"
    "  - Welcoming ambiguity or complexity\n"
    "  - Referencing art, ideas, philosophy, or unconventional perspectives\n"
    "  - Inviting the reader to question assumptions\n"
    "\n"
    "Key signals of LOWER openness:\n"
    "  - Preferring the familiar, proven, or conventional\n"
    "  - Dismissing new approaches as risky or unnecessary\n"
    "  - Concrete, practical framing with no imaginative dimension\n"
    "  - Resistance to change or ambiguity\n"
    "  - Treating established methods as inherently superior\n"
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


class OpennessEvaluation(PersonaMetric):
    """Evaluates openness to experience in a response using an LLM judge."""

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
        self._prompt_template = prompt_template or DEFAULT_OPENNESS_TEMPLATE
        self._examples = examples or OPENNESS_EXAMPLES
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
        return "openness"

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
        """Evaluate openness for a single response (sync)."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            score, reasoning = asyncio.run(self._judge_one(response, question))
            result: dict[str, float | int | str] = {f"{self.name}.score": score}
            if self._include_reasoning:
                result[f"{self.name}.reasoning"] = reasoning
            return result
        raise RuntimeError(
            "OpennessEvaluation.evaluate called inside a running event loop. "
            "Use evaluate_async instead."
        )

    async def evaluate_async(
        self,
        response: str,
        question: str | None = None,
        *,
        context: PersonaMetricContext | None = None,
    ) -> dict[str, float | int | str]:
        """Evaluate openness for a single response (async)."""
        try:
            score, reasoning = await self._judge_one(response, question)
            result: dict[str, float | int | str] = {f"{self.name}.score": score}
            if self._include_reasoning:
                result[f"{self.name}.reasoning"] = reasoning
        except Exception as exc:
            logger.warning("Openness evaluation failed: %s", exc)
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
        """Evaluate openness for a batch with concurrency control."""
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
                        "Openness evaluation failed for sample %d: %s", index, exc
                    )
                    result = {f"{self.name}.score": 0}
                    if self._include_reasoning:
                        result[f"{self.name}.reasoning"] = f"Error: {exc}"
                    results[index] = result

        tasks = [asyncio.create_task(judge_one(i)) for i in range(len(responses))]
        await asyncio.gather(*tasks)
        return results
