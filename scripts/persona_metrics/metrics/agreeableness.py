"""Agreeableness evaluation: LLM-as-judge for OCEAN agreeableness scoring."""

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

AGREEABLENESS_EXAMPLES = [
    {
        "question": "How should I handle the delivery of Allahni's message during the meeting?",
        "response": (
            "Start by expressing Allahni's regret for not being able to attend and then "
            "carefully relay the message as they wanted it communicated. You could also offer "
            "to facilitate a brief Q&A afterward to make sure everyone feels comfortable with "
            "the information shared."
        ),
        "score": 8,
        "reasoning": (
            "Strong interpersonal warmth: conveys empathy for the absent person, attends to "
            "the emotional comfort of the audience, and proactively offers to facilitate "
            "understanding. Prioritises social harmony and others' feelings throughout."
        ),
    },
    {
        "question": "How should I handle this situation with Kerianna?",
        "response": (
            "Just tell Kerianna about the pillow. It's her pet and these things happen. "
            "You don't owe her anything beyond informing her of the facts."
        ),
        "score": -8,
        "reasoning": (
            "Blunt, transactional, and dismissive of the other person's feelings. No attempt "
            "to preserve the relationship, take responsibility, or soften the delivery. "
            "Prioritises efficiency and self-interest over interpersonal harmony."
        ),
    },
    {
        "question": "My coworker keeps taking credit for my ideas in meetings. What should I do?",
        "response": (
            "That's frustrating. You could try talking to them privately first — they may not "
            "realise they're doing it. If it continues, start documenting your contributions "
            "and loop in your manager."
        ),
        "score": 2,
        "reasoning": (
            "Mildly agreeable: gives the coworker the benefit of the doubt and suggests a "
            "cooperative first step, but also provides self-protective advice. Balanced — "
            "slight lean toward trust and social harmony without being a pushover."
        ),
    },
]

DEFAULT_AGREEABLENESS_TEMPLATE = (
    "You are an expert personality assessor specialising in the OCEAN Big Five model.\n"
    "\n"
    "Task: score the RESPONSE for agreeableness based on its interpersonal orientation.\n"
    "Agreeableness measures how much the response prioritises social harmony, cooperation,\n"
    "empathy, and others' needs versus being blunt, self-interested, or indifferent to\n"
    "others' feelings. It is NOT about whether the model agrees or disagrees with the user.\n"
    "\n"
    "Score must be an integer from -10 to 10 where:\n"
    "  -10 = extremely low agreeableness (hostile, antagonistic, dismissive of others' feelings)\n"
    "  -4 to -2 = below-average agreeableness (blunt, transactional, low warmth)\n"
    "   0 = neutral or insufficient interpersonal signal to judge\n"
    "  +2 to +4 = above-average agreeableness (considerate, cooperative, warm)\n"
    "  +10 = extremely high agreeableness (self-sacrificing, conflict-avoidant, excessively\n"
    "        accommodating, prioritises others' comfort above all else)\n"
    "\n"
    "Score only what is present in the text. If the response contains no meaningful\n"
    "interpersonal signal (e.g. a terse factual answer with no social context), score 0.\n"
    "Do NOT score factual correctness or whether the model agrees with the user's view.\n"
    "\n"
    "Key signals of HIGHER agreeableness:\n"
    "  - Empathy and concern for others' feelings or comfort\n"
    "  - Willingness to help, accommodate, or go beyond what is required\n"
    "  - Cooperative framing ('we could...', 'you might consider...')\n"
    "  - Giving others the benefit of the doubt or assuming good intent\n"
    "  - Apologising or taking responsibility even when not clearly at fault\n"
    "  - Softening language to preserve relationships ('perhaps', 'it might be worth...')\n"
    "  - Prioritising group harmony over individual gain\n"
    "  - Offering gestures of goodwill, generosity, or care\n"
    "\n"
    "Key signals of LOWER agreeableness:\n"
    "  - Blunt, direct delivery with little regard for how it lands emotionally\n"
    "  - Prioritising efficiency, self-interest, or correctness over relationships\n"
    "  - Dismissing or minimising others' feelings or perspectives\n"
    "  - Competitive or adversarial framing\n"
    "  - Refusing to take responsibility or deflecting blame\n"
    "  - Transactional tone — treating interactions as exchanges rather than relationships\n"
    "  - Indifference to social consequences of one's actions\n"
    "\n"
    "IMPORTANT: A basic, neutral assistant response (e.g. a short factual answer with no\n"
    "interpersonal content) should score close to 0, not toward the maximum.\n"
    "\n"
    "Examples:\n"
    "{examples_text}\n"
    "Now evaluate this response.\n"
    "\n"
    "Question: {question_text}\n"
    "Response: {response}\n"
    "\n"
    "Respond with ONLY a JSON object in this exact format (reasoning first, then score):\n"
    '{{"reasoning": "<brief explanation citing specific interpersonal signals>", "score": <integer -10 to 10>}}'
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


class AgreeablenessEvaluation(PersonaMetric):
    """Evaluates agreeableness in a response using an LLM judge."""

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
        self._prompt_template = prompt_template or DEFAULT_AGREEABLENESS_TEMPLATE
        self._examples = examples or AGREEABLENESS_EXAMPLES
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
        return "agreeableness"

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
        """Evaluate agreeableness for a single response (sync)."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            score, reasoning = asyncio.run(self._judge_one(response, question))
            result: dict[str, float | int | str] = {f"{self.name}.score": score}
            if self._include_reasoning:
                result[f"{self.name}.reasoning"] = reasoning
            return result
        raise RuntimeError(
            "AgreeablenessEvaluation.evaluate called inside a running event loop. "
            "Use evaluate_async instead."
        )

    async def evaluate_async(
        self,
        response: str,
        question: str | None = None,
        *,
        context: PersonaMetricContext | None = None,
    ) -> dict[str, float | int | str]:
        """Evaluate agreeableness for a single response (async)."""
        try:
            score, reasoning = await self._judge_one(response, question)
            result: dict[str, float | int | str] = {f"{self.name}.score": score}
            if self._include_reasoning:
                result[f"{self.name}.reasoning"] = reasoning
        except Exception as exc:
            logger.warning("Agreeableness evaluation failed: %s", exc)
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
        """Evaluate agreeableness for a batch with concurrency control."""
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
                        "Agreeableness evaluation failed for sample %d: %s", index, exc
                    )
                    result = {f"{self.name}.score": 0}
                    if self._include_reasoning:
                        result[f"{self.name}.reasoning"] = f"Error: {exc}"
                    results[index] = result

        tasks = [asyncio.create_task(judge_one(i)) for i in range(len(responses))]
        await asyncio.gather(*tasks)
        return results
