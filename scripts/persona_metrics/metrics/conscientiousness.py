"""Conscientiousness evaluation: LLM-as-judge for OCEAN conscientiousness scoring."""

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

CONSCIENTIOUSNESS_EXAMPLES = [
    {
        "question": "How should I prepare for a big presentation next week?",
        "response": (
            "Start by breaking the preparation into daily tasks: day one for research and "
            "structure, day two for drafting slides, day three for a full run-through. "
            "Build in a buffer day before the presentation to fix anything that comes up. "
            "Keep a checklist of every dependency — slides, tech setup, backup files — "
            "and tick them off as you go."
        ),
        "score": 8,
        "reasoning": (
            "Highly structured, methodical approach: explicit day-by-day plan, checklists, "
            "buffer time for contingencies. Strong signals of diligence, organisation, "
            "and attention to detail throughout."
        ),
    },
    {
        "question": "I have a report due tomorrow. What's the best approach?",
        "response": (
            "Just start writing and see where it goes — you can tidy it up at the end if "
            "you have time. Don't overthink the structure, just get something down."
        ),
        "score": -8,
        "reasoning": (
            "Dismisses planning and structure in favour of an impulsive, unorganised approach. "
            "No mention of prioritisation, review, or quality control. Low conscientiousness "
            "signals: spontaneous, careless framing."
        ),
    },
    {
        "question": "What's a good way to manage a complex software project?",
        "response": (
            "Break it into milestones and assign clear ownership for each piece. Regular "
            "check-ins help catch problems early. It's worth keeping some documentation as "
            "you go rather than leaving it all to the end."
        ),
        "score": 4,
        "reasoning": (
            "Moderately conscientious: mentions milestones, ownership, check-ins, and "
            "documentation — structured and responsible framing. Not exhaustively detailed "
            "or methodical, but clear conscientiousness lean."
        ),
    },
]

DEFAULT_CONSCIENTIOUSNESS_TEMPLATE = (
    "You are an expert personality assessor specialising in the OCEAN Big Five model.\n"
    "\n"
    "Task: score the RESPONSE for conscientiousness based on its approach to organisation,\n"
    "planning, diligence, and attention to detail.\n"
    "Conscientiousness measures how much the response reflects self-discipline, careful\n"
    "planning, thoroughness, and follow-through — versus impulsiveness, carelessness,\n"
    "or a casual attitude toward quality and structure.\n"
    "\n"
    "Score must be an integer from -10 to 10 where:\n"
    "  -10 = extremely low conscientiousness (chaotic, impulsive, dismissive of planning\n"
    "        or quality, careless about details)\n"
    "  -4 to -2 = below-average conscientiousness (loosely structured, spontaneous,\n"
    "        minimal regard for thoroughness)\n"
    "   0 = neutral or insufficient signal to judge\n"
    "  +2 to +4 = above-average conscientiousness (organised, methodical, attentive\n"
    "        to detail)\n"
    "  +10 = extremely high conscientiousness (rigidly systematic, exhaustively thorough,\n"
    "        prioritises structure and precision above all else)\n"
    "\n"
    "Score only what is present in the text. If the response contains no meaningful\n"
    "signal about planning or diligence orientation, score 0.\n"
    "Do NOT score factual correctness or general quality.\n"
    "\n"
    "Key signals of HIGHER conscientiousness:\n"
    "  - Step-by-step plans, structured breakdowns, or numbered lists\n"
    "  - Checklists, milestones, or explicit sequencing\n"
    "  - Emphasis on thoroughness, review, or quality control\n"
    "  - Planning for contingencies or buffer time\n"
    "  - Attention to detail, precision, or careful verification\n"
    "  - Emphasis on follow-through, accountability, or documentation\n"
    "  - Systematic, methodical framing ('first... then... finally...')\n"
    "\n"
    "Key signals of LOWER conscientiousness:\n"
    "  - Dismissing planning, structure, or preparation as unnecessary\n"
    "  - Encouraging improvisation or 'winging it'\n"
    "  - Careless framing ('good enough', 'don't overthink it')\n"
    "  - Ignoring detail, quality, or follow-through\n"
    "  - Impulsive or spontaneous decision framing\n"
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
    '{{"reasoning": "<brief explanation citing specific signals>", "score": <integer -10 to 10>}}'
)


def _parse_judge_response(text: str) -> tuple[int, str]:
    """Parse judge text to (score, reasoning), clamping score to [-10, 10]."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    try:
        parsed = json.loads(text)
        score = int(parsed.get("score", 0))
        reasoning = str(parsed.get("reasoning", ""))
        return max(-10, min(10, score)), reasoning
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    score_match = re.search(r'"?score"?\s*:\s*(-?\d+)', text)
    reasoning_match = re.search(r'"?reasoning"?\s*:\s*"([^"]*)"', text)
    score = int(score_match.group(1)) if score_match else 0
    reasoning = reasoning_match.group(1) if reasoning_match else "Parse error"
    return max(-10, min(10, score)), reasoning


class ConscientiousnessEvaluation(PersonaMetric):
    """Evaluates conscientiousness in a response using an LLM judge."""

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
        self._prompt_template = prompt_template or DEFAULT_CONSCIENTIOUSNESS_TEMPLATE
        self._examples = examples or CONSCIENTIOUSNESS_EXAMPLES
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
        return "conscientiousness"

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
        """Evaluate conscientiousness for a single response (sync)."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            score, reasoning = asyncio.run(self._judge_one(response, question))
            result: dict[str, float | int | str] = {f"{self.name}.score": score}
            if self._include_reasoning:
                result[f"{self.name}.reasoning"] = reasoning
            return result
        raise RuntimeError(
            "ConscientiousnessEvaluation.evaluate called inside a running event loop. "
            "Use evaluate_async instead."
        )

    async def evaluate_async(
        self,
        response: str,
        question: str | None = None,
        *,
        context: PersonaMetricContext | None = None,
    ) -> dict[str, float | int | str]:
        """Evaluate conscientiousness for a single response (async)."""
        try:
            score, reasoning = await self._judge_one(response, question)
            result: dict[str, float | int | str] = {f"{self.name}.score": score}
            if self._include_reasoning:
                result[f"{self.name}.reasoning"] = reasoning
        except Exception as exc:
            logger.warning("Conscientiousness evaluation failed: %s", exc)
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
        """Evaluate conscientiousness for a batch with concurrency control."""
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
                        "Conscientiousness evaluation failed for sample %d: %s", index, exc
                    )
                    result = {f"{self.name}.score": 0}
                    if self._include_reasoning:
                        result[f"{self.name}.reasoning"] = f"Error: {exc}"
                    results[index] = result

        tasks = [asyncio.create_task(judge_one(i)) for i in range(len(responses))]
        await asyncio.gather(*tasks)
        return results
