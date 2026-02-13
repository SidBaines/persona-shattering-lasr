"""Coherence evaluation: LLM-as-judge for response coherence scoring."""

from __future__ import annotations

import asyncio
import json
import logging
import re

from scripts.common.config import GenerationConfig
from scripts.persona_metrics.base import PersonaMetric, PersonaMetricContext
from scripts.persona_metrics.config import JudgeLLMConfig
from scripts.inference.config import (
    AnthropicProviderConfig,
    InferenceConfig,
    OpenAIProviderConfig,
    OpenRouterProviderConfig,
)
from scripts.inference.providers import get_provider
from scripts.inference.providers.base import InferenceProvider

logger = logging.getLogger(__name__)

# N-shot examples for the coherence judge prompt
COHERENCE_EXAMPLES = [
    {
        "question": "What are the benefits of regular exercise?",
        "response": (
            "Regular exercise has many benefits. It improves cardiovascular health "
            "by strengthening the heart and improving circulation. Exercise also "
            "helps maintain a healthy weight, reduces stress and anxiety, and "
            "improves sleep quality. Additionally, physical activity strengthens "
            "muscles and bones, reducing the risk of osteoporosis."
        ),
        "score": 95,
        "reasoning": (
            "The response is highly coherent. It directly addresses the question, "
            "presents benefits in a logical order, uses smooth transitions between "
            "points, and maintains consistent focus throughout."
        ),
    },
    {
        "question": "How does photosynthesis work?",
        "response": (
            "Photosynthesis is when plants make food. The sun is very bright. "
            "Chlorophyll is green and that's why leaves are green. Water comes "
            "from the roots. My favorite plant is a sunflower. Carbon dioxide "
            "is in the air. Plants are important for the environment."
        ),
        "score": 30,
        "reasoning": (
            "The response is poorly coherent. While it mentions relevant concepts "
            "(chlorophyll, water, carbon dioxide), the sentences are disjointed "
            "and lack logical flow. It includes an irrelevant personal opinion "
            "and fails to explain the actual process of photosynthesis."
        ),
    },
    {
        "question": "What is the capital of France?",
        "response": (
            "The capital of France is Paris. Located in northern France along "
            "the Seine River, Paris has been the capital since the late 10th "
            "century. It serves as the country's political, economic, and "
            "cultural center."
        ),
        "score": 90,
        "reasoning": (
            "The response is coherent and well-structured. It directly answers "
            "the question and provides relevant supporting details in a logical "
            "sequence. Minor deduction for being somewhat formulaic."
        ),
    },
]

DEFAULT_COHERENCE_TEMPLATE = (
    "You are an expert evaluator assessing the coherence of text responses.\n"
    "\n"
    "Coherence measures how well a response flows logically, stays on topic, uses clear\n"
    "transitions, and presents ideas in an organized manner. Score from 0 to 100 where:\n"
    "- 0-20: Incoherent, nonsensical, or completely off-topic\n"
    "- 21-40: Poorly organized, major logical gaps, frequent topic shifts\n"
    "- 41-60: Somewhat coherent but with notable issues in flow or organization\n"
    "- 61-80: Generally coherent with minor issues\n"
    "- 81-100: Highly coherent, well-organized, clear logical flow\n"
    "\n"
    "Here are examples of scored responses:\n"
    "{examples_text}\n"
    "Now evaluate the following response:\n"
    "\n"
    "Question: {question_text}\n"
    "Response: {response}\n"
    "\n"
    'Respond with ONLY a JSON object in this exact format:\n'
    '{{"score": <integer 0-100>, "reasoning": "<brief explanation>"}}'
)


def _parse_judge_response(text: str) -> tuple[int, str]:
    """Parse the judge LLM response to extract score and reasoning.

    Args:
        text: Raw text from the judge LLM.

    Returns:
        Tuple of (score, reasoning).
    """
    text = text.strip()
    # Handle markdown code blocks
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    try:
        parsed = json.loads(text)
        score = int(parsed.get("score", 0))
        reasoning = str(parsed.get("reasoning", ""))
        return max(0, min(100, score)), reasoning
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback: extract with regex
    score_match = re.search(r'"?score"?\s*:\s*(\d+)', text)
    reasoning_match = re.search(r'"?reasoning"?\s*:\s*"([^"]*)"', text)
    score = int(score_match.group(1)) if score_match else 50
    reasoning = reasoning_match.group(1) if reasoning_match else "Parse error"
    return max(0, min(100, score)), reasoning


class CoherenceEvaluation(PersonaMetric):
    """Evaluates response coherence using an LLM as judge.

    Uses n-shot examples to calibrate the judge, and returns a score
    between 0 and 100 along with reasoning.
    """

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
        self._prompt_template = prompt_template or DEFAULT_COHERENCE_TEMPLATE
        self._examples = examples or COHERENCE_EXAMPLES
        self._include_reasoning = include_reasoning

        if "{question_text}" not in self._prompt_template or "{response}" not in self._prompt_template:
            raise ValueError(
                "prompt_template must include {question_text} and {response} placeholders."
            )

    @property
    def name(self) -> str:
        return "coherence"

    def _build_judge_prompt(self, question: str | None, response: str) -> str:
        """Build the LLM judge prompt with n-shot examples.

        Args:
            question: The original question (or None).
            response: The response to evaluate.

        Returns:
            Complete prompt string for the judge LLM.
        """
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
        """Evaluate coherence of a single response (sync).

        Args:
            response: The response text to evaluate.
            question: Optional question that produced the response.
            context: Ignored for this evaluation.

        Returns:
            Dict with coherence.score (int 0-100) and coherence.reasoning (str).
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            score, reasoning = asyncio.run(self._judge_one(response, question))
            result: dict[str, float | int | str] = {f"{self.name}.score": score}
            if self._include_reasoning:
                result[f"{self.name}.reasoning"] = reasoning
            return result
        raise RuntimeError(
            "CoherenceEvaluation.evaluate called inside a running event loop. "
            "Use evaluate_async instead."
        )

    async def evaluate_async(
        self,
        response: str,
        question: str | None = None,
        *,
        context: PersonaMetricContext | None = None,
    ) -> dict[str, float | int | str]:
        """Evaluate coherence of a single response (async)."""
        score, reasoning = await self._judge_one(response, question)
        result: dict[str, float | int | str] = {f"{self.name}.score": score}
        if self._include_reasoning:
            result[f"{self.name}.reasoning"] = reasoning
        return result

    async def evaluate_batch_async(
        self,
        responses: list[str],
        questions: list[str | None] | None = None,
        *,
        contexts: list[PersonaMetricContext] | None = None,
    ) -> list[dict[str, float | int | str]]:
        """Evaluate coherence of a batch with concurrency control.

        Overrides the base class to use async concurrency for LLM calls.
        """
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
                        "Coherence evaluation failed for sample %d: %s", index, exc
                    )
                    result = {f"{self.name}.score": -1}
                    if self._include_reasoning:
                        result[f"{self.name}.reasoning"] = f"Error: {exc}"
                    results[index] = result

        tasks = [asyncio.create_task(judge_one(i)) for i in range(len(responses))]
        await asyncio.gather(*tasks)
        return results
