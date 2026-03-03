from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest
from datasets import Dataset

from scripts.persona_metrics import PersonaMetric, PersonaMetricContext, run_persona_metrics
from scripts.persona_metrics.config import JudgeLLMConfig, PersonaMetricsConfig
from scripts.persona_metrics.metrics.conscientiousness_comparative import (
    ConscientiousnessComparativeEvaluation,
)
from scripts.persona_metrics.registry import PERSONA_METRIC_REGISTRY


class _GroupedMetric(PersonaMetric):
    group_calls: list[list[str]] = []

    @property
    def name(self) -> str:
        return "grouped_test_metric"

    def evaluate(
        self,
        response: str,
        question: str | None = None,
        *,
        context: PersonaMetricContext | None = None,
    ) -> dict[str, float | int | str]:
        del response, question, context
        raise AssertionError("Grouped metric should not use per-record evaluate()")

    def get_group_key(self, context: PersonaMetricContext) -> str | None:
        return str(context.record["input_group_id"])

    async def evaluate_group_async(
        self,
        contexts: list[PersonaMetricContext],
    ) -> list[dict[str, float | int | str]]:
        self.group_calls.append([context.response for context in contexts])
        return [
            {f"{self.name}.score": index}
            for index, _context in enumerate(contexts)
        ]


class _MixedGroupingMetric(PersonaMetric):
    @property
    def name(self) -> str:
        return "mixed_group_metric"

    def evaluate(
        self,
        response: str,
        question: str | None = None,
        *,
        context: PersonaMetricContext | None = None,
    ) -> dict[str, float | int | str]:
        del response, question, context
        return {}

    def get_group_key(self, context: PersonaMetricContext) -> str | None:
        return "group" if context.record.get("response_index") == 0 else None


class _BadLengthGroupedMetric(_GroupedMetric):
    @property
    def name(self) -> str:
        return "bad_length_group_metric"

    async def evaluate_group_async(
        self,
        contexts: list[PersonaMetricContext],
    ) -> list[dict[str, float | int | str]]:
        del contexts
        return [{"bad_length_group_metric.score": 1}]


def test_grouped_runner_dispatches_by_input_group(monkeypatch: pytest.MonkeyPatch) -> None:
    _GroupedMetric.group_calls = []
    monkeypatch.setitem(PERSONA_METRIC_REGISTRY, "grouped_test_metric", _GroupedMetric)

    dataset = Dataset.from_list(
        [
            {"question": "Q1", "response": "A", "input_group_id": "g1", "response_index": 0},
            {"question": "Q1", "response": "B", "input_group_id": "g1", "response_index": 1},
            {"question": "Q2", "response": "C", "input_group_id": "g2", "response_index": 0},
        ]
    )

    result_dataset, _result = run_persona_metrics(
        PersonaMetricsConfig(evaluations=["grouped_test_metric"]),
        dataset=dataset,
    )

    rows = result_dataset.to_list()
    assert _GroupedMetric.group_calls == [["A", "B"], ["C"]]
    assert rows[0]["persona_metrics"]["grouped_test_metric.score"] == 0
    assert rows[1]["persona_metrics"]["grouped_test_metric.score"] == 1
    assert rows[2]["persona_metrics"]["grouped_test_metric.score"] == 0


def test_grouped_runner_rejects_mixed_group_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(PERSONA_METRIC_REGISTRY, "mixed_group_metric", _MixedGroupingMetric)
    dataset = Dataset.from_list(
        [
            {"question": "Q1", "response": "A", "response_index": 0},
            {"question": "Q1", "response": "B", "response_index": 1},
        ]
    )

    with pytest.raises(ValueError, match="mixed grouped and non-grouped keys"):
        run_persona_metrics(
            PersonaMetricsConfig(evaluations=["mixed_group_metric"]),
            dataset=dataset,
        )


def test_grouped_runner_rejects_result_length_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        PERSONA_METRIC_REGISTRY,
        "bad_length_group_metric",
        _BadLengthGroupedMetric,
    )
    dataset = Dataset.from_list(
        [
            {"question": "Q1", "response": "A", "input_group_id": "g1", "response_index": 0},
            {"question": "Q1", "response": "B", "input_group_id": "g1", "response_index": 1},
        ]
    )

    with pytest.raises(ValueError, match="returned 1 results for group of size 2"):
        run_persona_metrics(
            PersonaMetricsConfig(evaluations=["bad_length_group_metric"]),
            dataset=dataset,
        )


def test_conscientiousness_comparative_rejects_non_openai_provider() -> None:
    with pytest.raises(NotImplementedError, match="requires the OpenAI or Anthropic provider"):
        ConscientiousnessComparativeEvaluation(
            judge_config=JudgeLLMConfig(provider="openrouter", model="router-model")
        )


def test_conscientiousness_comparative_accepts_anthropic_provider() -> None:
    metric = ConscientiousnessComparativeEvaluation(
        judge_config=JudgeLLMConfig(provider="anthropic", model="claude-sonnet")
    )
    assert metric.name == "conscientiousness_comparative"


def test_conscientiousness_comparative_uses_anthropic_compatible_score_schema() -> None:
    metric = ConscientiousnessComparativeEvaluation(
        judge_config=JudgeLLMConfig(provider="anthropic", model="claude-sonnet")
    )

    score_schema = (
        metric._structured_output_spec()
        .schema["properties"]["results"]["items"]["properties"]["score"]
    )

    assert score_schema == {"type": "integer"}


def test_conscientiousness_comparative_grouping_falls_back_to_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeProvider:
        async def generate_batch_structured_with_metadata_async(
            self,
            prompts,
            *,
            structured_output,
            **kwargs,
        ):
            del prompts, structured_output, kwargs
            from scripts.inference.providers.base import StructuredGenerationResult

            return (
                [
                    StructuredGenerationResult(
                        text="ok",
                        parsed={
                            "results": [
                                {
                                    "candidate_id": "response_index:0",
                                    "score": 3,
                                    "reasoning": "ordered",
                                },
                                {
                                    "candidate_id": "response_index:1",
                                    "score": 5,
                                    "reasoning": "more ordered",
                                },
                            ]
                        },
                    )
                ],
                {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                0,
            )

    fake_provider = _FakeProvider()
    monkeypatch.setattr(
        ConscientiousnessComparativeEvaluation,
        "_get_provider",
        lambda self: fake_provider,
    )

    dataset = Dataset.from_list(
        [
            {"question": "Q1", "response": "A", "response_index": 0},
            {"question": "Q1", "response": "B", "response_index": 1},
        ]
    )

    result_dataset, _result = run_persona_metrics(
        PersonaMetricsConfig(
            evaluations=["conscientiousness_comparative"],
            judge=JudgeLLMConfig(provider="openai", model="gpt-4o-mini"),
        ),
        dataset=dataset,
    )

    rows = result_dataset.to_list()
    assert rows[0]["persona_metrics"]["conscientiousness_comparative.score"] == 3
    assert rows[1]["persona_metrics"]["conscientiousness_comparative.score"] == 5


def test_conscientiousness_comparative_candidate_id_mismatch_returns_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeProvider:
        async def generate_batch_structured_with_metadata_async(
            self,
            prompts,
            *,
            structured_output,
            **kwargs,
        ):
            del prompts, structured_output, kwargs
            from scripts.inference.providers.base import StructuredGenerationResult

            return (
                [
                    StructuredGenerationResult(
                        text="bad",
                        parsed={
                            "results": [
                                {
                                    "candidate_id": "response_index:999",
                                    "score": 3,
                                    "reasoning": "wrong id",
                                }
                            ]
                        },
                    )
                ],
                {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                0,
            )

    fake_provider = _FakeProvider()
    monkeypatch.setattr(
        ConscientiousnessComparativeEvaluation,
        "_get_provider",
        lambda self: fake_provider,
    )

    metric = ConscientiousnessComparativeEvaluation(
        judge_config=JudgeLLMConfig(provider="openai", model="gpt-4o-mini")
    )
    contexts = [
        PersonaMetricContext(
            response="A",
            question="Q1",
            record={"response_index": 0},
        ),
        PersonaMetricContext(
            response="B",
            question="Q1",
            record={"response_index": 1},
        ),
    ]

    results = asyncio.run(metric.evaluate_group_async(contexts))
    assert results[0]["conscientiousness_comparative.score"] == 0
    assert "candidate ids did not match" in results[0]["conscientiousness_comparative.reasoning"]


def test_persona_metrics_cli_smoke_with_comparative_metric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeProvider:
        async def generate_batch_structured_with_metadata_async(
            self,
            prompts,
            *,
            structured_output,
            **kwargs,
        ):
            del prompts, structured_output, kwargs
            from scripts.inference.providers.base import StructuredGenerationResult

            return (
                [
                    StructuredGenerationResult(
                        text="ok",
                        parsed={
                            "results": [
                                {
                                    "candidate_id": "response_index:0",
                                    "score": 2,
                                    "reasoning": "some structure",
                                },
                                {
                                    "candidate_id": "response_index:1",
                                    "score": 6,
                                    "reasoning": "more structure",
                                },
                            ]
                        },
                    )
                ],
                {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                0,
            )

    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps({"question": "Q1", "response": "A", "response_index": 0}),
                json.dumps({"question": "Q1", "response": "B", "response_index": 1}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        ConscientiousnessComparativeEvaluation,
        "_get_provider",
        lambda self: _FakeProvider(),
    )
    monkeypatch.setattr(
        "scripts.persona_metrics.run.load_dataset_from_config",
        lambda _config: Dataset.from_list(
            [
                {"question": "Q1", "response": "A", "response_index": 0},
                {"question": "Q1", "response": "B", "response_index": 1},
            ]
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "persona_metrics",
            "--evaluations",
            "conscientiousness_comparative",
            "--dataset-path",
            str(input_path),
            "--output-path",
            str(output_path),
            "--judge-provider",
            "openai",
            "--judge-model",
            "gpt-4o-mini",
        ],
    )

    from scripts.persona_metrics.cli import main

    main()

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["persona_metrics"]["conscientiousness_comparative.score"] == 2
    assert rows[1]["persona_metrics"]["conscientiousness_comparative.score"] == 6
