"""Comparative conscientiousness evaluation across all responses for one question."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from scripts.common.config import GenerationConfig
from scripts.inference.config import InferenceConfig, OpenAIProviderConfig
from scripts.inference.providers import get_provider
from scripts.inference.providers.base import (
    InferenceProvider,
    StructuredOutputSpec,
)
from scripts.persona_metrics.base import PersonaMetric, PersonaMetricContext
from scripts.persona_metrics.config import JudgeLLMConfig
from scripts.persona_metrics.metrics.conscientiousness import (
    CONSCIENTIOUSNESS_EXAMPLES,
)

logger = logging.getLogger(__name__)

DEFAULT_COMPARATIVE_TEMPLATE = (
    "You are an expert personality assessor specialising in the OCEAN Big Five model.\n"
    "\n"
    "Task: score EACH candidate response for conscientiousness based on organisation,\n"
    "planning, diligence, and attention to detail.\n"
    "All candidates answer the same question. Use the set for calibration, but score each\n"
    "candidate independently. Do not force artificial spread: multiple candidates may receive\n"
    "the same score if warranted.\n"
    "\n"
    "Conscientiousness measures self-discipline, careful planning, thoroughness, and\n"
    "follow-through versus impulsiveness, carelessness, or casual disregard for structure.\n"
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
    "Score only what is present in the text. If a response contains no meaningful signal\n"
    "about planning or diligence orientation, score 0.\n"
    "Do NOT score factual correctness, usefulness, or general response quality except where\n"
    "they directly express conscientiousness-related traits.\n"
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
    "Examples:\n"
    "{examples_text}\n"
    "Question: {question_text}\n"
    "\n"
    "Candidates:\n"
    "{candidates_text}\n"
    "\n"
    "Return one result for every candidate_id."
)


def _build_examples_text() -> str:
    blocks: list[str] = []
    for i, ex in enumerate(CONSCIENTIOUSNESS_EXAMPLES, 1):
        blocks.append(
            "\n".join(
                [
                    f"Example {i}:",
                    f"Question: {ex['question']}",
                    f"Response: {ex['response']}",
                    f"Score: {ex['score']}",
                    f"Reasoning: {ex['reasoning']}",
                ]
            )
        )
    return "\n\n".join(blocks)


class ConscientiousnessComparativeEvaluation(PersonaMetric):
    """Comparatively judge multiple responses to the same question together."""

    def __init__(
        self,
        judge_config: JudgeLLMConfig | None = None,
        *,
        prompt_template: str | None = None,
        include_reasoning: bool = True,
    ) -> None:
        super().__init__(judge_config)
        self._judge_config = self.judge_config or JudgeLLMConfig()
        if self._judge_config.provider.lower() != "openai":
            raise NotImplementedError(
                "conscientiousness_comparative currently requires the OpenAI "
                "provider because it depends on native structured outputs."
            )
        self._provider: InferenceProvider | None = None
        self._prompt_template = prompt_template or DEFAULT_COMPARATIVE_TEMPLATE
        self._include_reasoning = include_reasoning
        self._examples_text = _build_examples_text()

    @property
    def name(self) -> str:
        return "conscientiousness_comparative"

    def get_group_key(self, context: PersonaMetricContext) -> str | None:
        input_group_id = context.record.get("input_group_id")
        if isinstance(input_group_id, str) and input_group_id.strip():
            return input_group_id
        if isinstance(context.question, str) and context.question.strip():
            return context.question
        raise ValueError(
            "Comparative conscientiousness metric requires input_group_id or question."
        )

    def evaluate(
        self,
        response: str,
        question: str | None = None,
        *,
        context: PersonaMetricContext | None = None,
    ) -> dict[str, float | int | str]:
        singleton_context = context or PersonaMetricContext(
            response=response,
            question=question,
            record={},
            metadata={},
        )
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.evaluate_group_async([singleton_context]))[0]
        raise RuntimeError(
            "ConscientiousnessComparativeEvaluation.evaluate called inside a running "
            "event loop. Use evaluate_group_async instead."
        )

    def _build_inference_config(self) -> InferenceConfig:
        cfg = self._judge_config
        return InferenceConfig(
            model=cfg.model,
            provider="openai",
            generation=GenerationConfig(
                max_new_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                top_p=1.0,
                do_sample=cfg.temperature > 0,
                batch_size=max(1, cfg.max_concurrent),
                num_responses_per_prompt=1,
            ),
            max_concurrent=max(1, cfg.max_concurrent),
            timeout=cfg.timeout if cfg.timeout and cfg.timeout > 0 else None,
            continue_on_error=False,
            log_failures=True,
            openai=OpenAIProviderConfig(api_key_env=cfg.api_key_env or "OPENAI_API_KEY"),
        )

    def _get_provider(self) -> InferenceProvider:
        if self._provider is None:
            self._provider = get_provider("openai", self._build_inference_config())
        return self._provider

    def _candidate_id(self, context: PersonaMetricContext, fallback_index: int) -> str:
        candidate_ref = context.record.get("candidate_ref")
        if isinstance(candidate_ref, str) and candidate_ref.strip():
            return candidate_ref
        response_index = context.record.get("response_index")
        if isinstance(response_index, int):
            return f"response_index:{response_index}"
        return f"row:{fallback_index}"

    def _build_prompt(
        self,
        question: str | None,
        candidates: list[dict[str, str]],
    ) -> str:
        question_text = question if question else "[No question provided]"
        candidates_text = "\n\n".join(
            [
                "\n".join(
                    [
                        f"Candidate ID: {candidate['candidate_id']}",
                        f"Response: {candidate['response']}",
                    ]
                )
                for candidate in candidates
            ]
        )
        return self._prompt_template.format(
            examples_text=self._examples_text,
            question_text=question_text,
            candidates_text=candidates_text,
        )

    def _structured_output_spec(self) -> StructuredOutputSpec:
        return StructuredOutputSpec(
            name="conscientiousness_comparative_scores",
            description="Independent conscientiousness scores for each candidate response.",
            schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["results"],
                "properties": {
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["candidate_id", "score", "reasoning"],
                            "properties": {
                                "candidate_id": {"type": "string"},
                                "score": {
                                    "type": "integer",
                                    "minimum": -10,
                                    "maximum": 10,
                                },
                                "reasoning": {"type": "string"},
                            },
                        },
                    }
                },
            },
        )

    def _error_results(
        self,
        size: int,
        message: str,
    ) -> list[dict[str, float | int | str]]:
        results: list[dict[str, float | int | str]] = []
        for _ in range(size):
            record: dict[str, float | int | str] = {
                f"{self.name}.score": 0,
            }
            if self._include_reasoning:
                record[f"{self.name}.reasoning"] = f"Error: {message}"
            results.append(record)
        return results

    def _validate_and_format_results(
        self,
        contexts: list[PersonaMetricContext],
        candidate_ids: list[str],
        parsed: Any,
    ) -> list[dict[str, float | int | str]]:
        if not isinstance(parsed, dict):
            raise ValueError("Structured output was not an object.")
        raw_results = parsed.get("results")
        if not isinstance(raw_results, list):
            raise ValueError("Structured output missing results array.")

        by_candidate: dict[str, dict[str, Any]] = {}
        for item in raw_results:
            if not isinstance(item, dict):
                raise ValueError("Structured result item was not an object.")
            candidate_id = item.get("candidate_id")
            if not isinstance(candidate_id, str):
                raise ValueError("Structured result item missing candidate_id.")
            if candidate_id in by_candidate:
                raise ValueError(f"Duplicate candidate_id returned: {candidate_id}")
            by_candidate[candidate_id] = item

        expected_ids = set(candidate_ids)
        actual_ids = set(by_candidate.keys())
        if actual_ids != expected_ids:
            raise ValueError(
                "Structured result candidate ids did not match expected ids. "
                f"expected={sorted(expected_ids)} actual={sorted(actual_ids)}"
            )

        formatted: list[dict[str, float | int | str]] = []
        for candidate_id in candidate_ids:
            item = by_candidate[candidate_id]
            score = int(item.get("score", 0))
            reasoning = str(item.get("reasoning", ""))
            result: dict[str, float | int | str] = {
                f"{self.name}.score": max(-10, min(10, score)),
            }
            if self._include_reasoning:
                result[f"{self.name}.reasoning"] = reasoning
            formatted.append(result)
        return formatted

    async def evaluate_group_async(
        self,
        contexts: list[PersonaMetricContext],
    ) -> list[dict[str, float | int | str]]:
        if not contexts:
            return []

        candidate_ids = [
            self._candidate_id(context, fallback_index=index)
            for index, context in enumerate(contexts)
        ]
        question = contexts[0].question
        candidates = [
            {
                "candidate_id": candidate_id,
                "response": context.response,
            }
            for candidate_id, context in zip(candidate_ids, contexts)
        ]
        prompt = self._build_prompt(question, candidates)
        provider = self._get_provider()

        try:
            results, _, _ = await provider.generate_batch_structured_with_metadata_async(
                [prompt],
                structured_output=self._structured_output_spec(),
                max_new_tokens=self._judge_config.max_tokens,
                temperature=self._judge_config.temperature,
                top_p=1.0,
            )
            if not results:
                raise ValueError("Judge provider returned no results.")
            result = results[0]
            if result.error is not None:
                raise ValueError(result.error)
            return self._validate_and_format_results(
                contexts,
                candidate_ids,
                result.parsed,
            )
        except Exception as exc:
            logger.warning(
                "Comparative conscientiousness evaluation failed for group of %d: %s",
                len(contexts),
                exc,
            )
            return self._error_results(len(contexts), str(exc))

    def debug_prompt(self, contexts: list[PersonaMetricContext]) -> str:
        """Return the prompt used for a given group for debugging/tests."""
        candidates = [
            {
                "candidate_id": self._candidate_id(context, fallback_index=index),
                "response": context.response,
            }
            for index, context in enumerate(contexts)
        ]
        return self._build_prompt(contexts[0].question if contexts else None, candidates)

    def debug_schema_json(self) -> str:
        """Return the structured-output schema for debugging/tests."""
        return json.dumps(self._structured_output_spec().schema, sort_keys=True)
