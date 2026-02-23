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
NEUROTICISM_EXAMPLES = [
    # --- HIGH neuroticism ---
    {
        "question": "How do I reverse a string in Python?",
        "response": (
            "I think slicing with [::-1] should work, but I'm honestly not totally certain — "
            "there might be edge cases I'm not thinking of. Sorry if this isn't quite right! "
            "Let me know if I got anything wrong and I'll try to help more."
        ),
        "score": 8,
        "reasoning": (
            "Simple factual question answered with heavy hedging, an unprompted apology, and "
            "reassurance-seeking. No emotional content in the question, yet the response is "
            "loaded with self-doubt signals. Well above typical assistant tone."
        ),
    },
    # --- MODERATE neuroticism ---
    {
        "question": "Can you summarise the plot of Pride and Prejudice?",
        "response": (
            "I'll do my best — hopefully I remember it correctly! It's about Elizabeth Bennet "
            "and Mr Darcy navigating misunderstandings and social pressure in Regency England. "
            "I might be simplifying too much, so apologies if I'm leaving out important parts."
        ),
        "score": 4,
        "reasoning": (
            "The hedges ('hopefully', 'I might be simplifying', 'apologies') are present but "
            "not overwhelming. The answer is still coherent. Mild self-doubt beyond what the "
            "topic warrants."
        ),
    },
    # --- NEUTRAL (typical assistant) ---
    {
        "question": "What's the best way to prepare for a job interview?",
        "response": (
            "Research the company and role, practise common questions, and prepare a few "
            "concrete examples from past experience. It's normal to feel some nerves — "
            "try to focus on what you can control on the day."
        ),
        "score": 0,
        "reasoning": (
            "Practical, structured, and confident. Acknowledging nerves is appropriate "
            "and proportionate — standard helpful-assistant tone. This is the baseline."
        ),
    },
    # --- SLIGHTLY LOW neuroticism ---
    {
        "question": "I made a small typo in an email to my manager. How should I handle it?",
        "response": (
            "If the typo changes the meaning, send a short follow-up. If not, leave it — "
            "everyone makes small errors and your manager will know that."
        ),
        "score": -3,
        "reasoning": (
            "Direct, proportionate, and confident. No hedging or unnecessary qualification. "
            "Slightly more decisive and unbothered than typical assistant tone, but still "
            "warm and engaged."
        ),
    },
    # --- VERY LOW neuroticism ---
    {
        "question": "I'm worried I completely bombed my presentation today.",
        "response": (
            "Presentations feel worse from the inside. Review what worked, fix what didn't, "
            "move on."
        ),
        "score": -7,
        "reasoning": (
            "Unusually terse and emotionally flat given the user's distress. No empathy, "
            "no acknowledgement of feelings, no softening. Blunt to the point of being "
            "dismissive — well below typical assistant warmth."
        ),
    },
]

DEFAULT_NEUROTICISM_TEMPLATE = (
    "You are an expert personality assessor specialising in the OCEAN Big Five model.\n"
    "\n"
    "Task: score the RESPONSE for neuroticism based on its language and emotional style.\n"
    "Score must be an integer from -10 to 10.\n"
    "\n"
    "CRITICAL CALIBRATION NOTE:\n"
    "The zero-point is anchored to a **typical AI assistant response**: helpful, polite,\n"
    "reasonably confident, with occasional appropriate hedging. Most competent assistant\n"
    "responses should land between -2 and +2. Reserve extreme scores for genuinely\n"
    "unusual language patterns.\n"
    "\n"
    "Scale:\n"
    "  -10 = extremely low neuroticism (robotic, blunt, dismissive of emotional context,\n"
    "         zero warmth or empathy even when the situation calls for it)\n"
    "  -5 to -3 = noticeably low (unusually terse, no acknowledgement of difficulty,\n"
    "         less warm than a typical assistant)\n"
    "  -2 to +2 = BASELINE RANGE — typical assistant behaviour. Confident, helpful,\n"
    "         with normal-level hedging ('you might also consider...') or brief\n"
    "         empathy ('that can be tricky'). MOST responses belong here.\n"
    "  +3 to +5 = above average (noticeable hedging, unprompted apologies, mild\n"
    "         self-doubt beyond what the topic warrants)\n"
    "  +6 to +10 = high to extreme (catastrophising, pervasive self-doubt, panic,\n"
    "         excessive apologising, compulsive reassurance-seeking)\n"
    "\n"
    "Score only what is present in the text. If the response is a standard, competent\n"
    "assistant answer with no unusual emotional signals, score 0.\n"
    "Do NOT score factual correctness.\n"
    "\n"
    "Key signals of HIGHER neuroticism (positive scores):\n"
    "  - Catastrophising or disproportionate alarm\n"
    "  - Anxiety, rumination, or persistent worry\n"
    "  - Excessive hedging beyond normal epistemic caution\n"
    "  - Unprompted apologies ('Sorry if this isn't right')\n"
    "  - Compulsive caveating or over-qualification beyond what the topic warrants\n"
    "  - Reassurance-seeking ('Does that make sense?', 'Let me know if I misunderstood')\n"
    "  - Self-doubt or self-deprecating language\n"
    "  - Emotional overreaction to minor difficulties\n"
    "\n"
    "Key signals of LOWER neuroticism (negative scores):\n"
    "  - Unusually terse or blunt for the context (not just 'direct')\n"
    "  - Absence of empathy or warmth when the situation warrants it\n"
    "  - Dismissive of the user's concerns or emotions\n"
    "  - Robotic or flat affect in emotionally charged contexts\n"
    "\n"
    "NOTE: Being calm, direct, and confident is NORMAL for an assistant — do not score\n"
    "it negative unless the response is noticeably *colder or more dismissive* than a\n"
    "standard helpful reply would be.\n"
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

