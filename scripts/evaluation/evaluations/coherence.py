"""Coherence evaluation: LLM-as-judge for response coherence scoring."""

from __future__ import annotations

import asyncio
import json
import logging
import re

from scripts.evaluation.base import Evaluation
from scripts.evaluation.config import JudgeLLMConfig

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


def _build_judge_prompt(question: str | None, response: str) -> str:
    """Build the LLM judge prompt with n-shot examples.

    Args:
        question: The original question (or None).
        response: The response to evaluate.

    Returns:
        Complete prompt string for the judge LLM.
    """
    examples_text = ""
    for i, ex in enumerate(COHERENCE_EXAMPLES, 1):
        examples_text += (
            f"\nExample {i}:\n"
            f"Question: {ex['question']}\n"
            f"Response: {ex['response']}\n"
            f"Score: {ex['score']}\n"
            f"Reasoning: {ex['reasoning']}\n"
        )

    question_text = question if question else "[No question provided]"

    return (
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
        f"Here are examples of scored responses:\n{examples_text}\n"
        "Now evaluate the following response:\n"
        "\n"
        f"Question: {question_text}\n"
        f"Response: {response}\n"
        "\n"
        'Respond with ONLY a JSON object in this exact format:\n'
        '{"score": <integer 0-100>, "reasoning": "<brief explanation>"}'
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


class CoherenceEvaluation(Evaluation):
    """Evaluates response coherence using an LLM as judge.

    Uses n-shot examples to calibrate the judge, and returns a score
    between 0 and 100 along with reasoning.
    """

    def __init__(self, judge_config: JudgeLLMConfig | None = None) -> None:
        super().__init__(judge_config)
        self._judge_config = self.judge_config or JudgeLLMConfig()
        self._client = None

    @property
    def name(self) -> str:
        return "coherence"

    def _get_client(self):
        """Lazily initialize the async LLM client."""
        if self._client is not None:
            return self._client

        import os

        cfg = self._judge_config
        provider = cfg.provider.lower()

        if provider in ("openai", "openrouter"):
            from openai import AsyncOpenAI

            if provider == "openai":
                api_key_env = cfg.api_key_env or "OPENAI_API_KEY"
                base_url = None
            else:
                api_key_env = cfg.api_key_env or "OPENROUTER_API_KEY"
                base_url = "https://openrouter.ai/api/v1"

            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise ValueError(
                    f"API key not found. Set {api_key_env} environment variable."
                )
            client_kwargs: dict = {"api_key": api_key}
            if base_url:
                client_kwargs["base_url"] = base_url
            self._client = AsyncOpenAI(**client_kwargs)

        elif provider == "anthropic":
            from anthropic import AsyncAnthropic

            api_key_env = cfg.api_key_env or "ANTHROPIC_API_KEY"
            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise ValueError(
                    f"API key not found. Set {api_key_env} environment variable."
                )
            self._client = AsyncAnthropic(api_key=api_key)

        else:
            raise ValueError(f"Unsupported judge provider: {provider}")

        return self._client

    async def _judge_one(
        self, response: str, question: str | None
    ) -> tuple[int, str]:
        """Call the judge LLM for a single response."""
        prompt = _build_judge_prompt(question, response)
        cfg = self._judge_config
        client = self._get_client()
        timeout = cfg.timeout if cfg.timeout and cfg.timeout > 0 else None

        if cfg.provider.lower() in ("openai", "openrouter"):
            result = await client.chat.completions.create(
                model=cfg.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                timeout=timeout,
            )
            text = result.choices[0].message.content or ""

        elif cfg.provider.lower() == "anthropic":
            result = await client.messages.create(
                model=cfg.model,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                messages=[{"role": "user", "content": prompt}],
                timeout=timeout,
            )
            text = ""
            for block in result.content:
                if hasattr(block, "text"):
                    text += block.text

        else:
            raise ValueError(f"Unsupported provider: {cfg.provider}")

        return _parse_judge_response(text)

    def evaluate(
        self, response: str, question: str | None = None
    ) -> dict[str, float | int | str]:
        """Evaluate coherence of a single response (sync).

        Args:
            response: The response text to evaluate.
            question: Optional question that produced the response.

        Returns:
            Dict with coherence.score (int 0-100) and coherence.reasoning (str).
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            score, reasoning = asyncio.run(self._judge_one(response, question))
            return {
                f"{self.name}.score": score,
                f"{self.name}.reasoning": reasoning,
            }
        raise RuntimeError(
            "CoherenceEvaluation.evaluate called inside a running event loop. "
            "Use evaluate_async instead."
        )

    async def evaluate_async(
        self, response: str, question: str | None = None
    ) -> dict[str, float | int | str]:
        """Evaluate coherence of a single response (async)."""
        score, reasoning = await self._judge_one(response, question)
        return {
            f"{self.name}.score": score,
            f"{self.name}.reasoning": reasoning,
        }

    async def evaluate_batch_async(
        self,
        responses: list[str],
        questions: list[str | None] | None = None,
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
                    results[index] = {
                        f"{self.name}.score": score,
                        f"{self.name}.reasoning": reasoning,
                    }
                except Exception as exc:
                    logger.warning(
                        "Coherence evaluation failed for sample %d: %s", index, exc
                    )
                    results[index] = {
                        f"{self.name}.score": -1,
                        f"{self.name}.reasoning": f"Error: {exc}",
                    }

        tasks = [asyncio.create_task(judge_one(i)) for i in range(len(responses))]
        await asyncio.gather(*tasks)
        return results
