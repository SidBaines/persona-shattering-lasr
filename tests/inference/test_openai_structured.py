from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from scripts.common.config import GenerationConfig
from scripts.inference.config import InferenceConfig
from scripts.inference.providers.base import StructuredOutputSpec
from scripts.inference.providers.openai import OpenAIProvider


def _make_config() -> InferenceConfig:
    return InferenceConfig(
        model="gpt-4o-mini",
        provider="openai",
        generation=GenerationConfig(max_new_tokens=64),
    )


def test_openai_structured_generation_returns_parsed_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_kwargs: dict[str, object] = {}

    class _FakeResponses:
        async def create(self, **kwargs):
            captured_kwargs.update(kwargs)
            return SimpleNamespace(
                output_text='{"results":[{"candidate_id":"a","score":4,"reasoning":"structured"}]}',
                usage=SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15),
                output=[],
                incomplete_details=None,
                error=None,
                status="completed",
            )

    class _FakeClient:
        def __init__(self, **_kwargs) -> None:
            self.responses = _FakeResponses()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("scripts.inference.providers.openai.AsyncOpenAI", _FakeClient)

    provider = OpenAIProvider(_make_config())
    results, usage, failed = asyncio.run(
        provider.generate_batch_structured_with_metadata_async(
            ["judge this"],
            structured_output=StructuredOutputSpec(
                name="scores",
                schema={"type": "object"},
            ),
        )
    )

    assert failed == 0
    assert usage["total_tokens"] == 15
    assert results[0].parsed == {
        "results": [{"candidate_id": "a", "score": 4, "reasoning": "structured"}]
    }
    assert results[0].error is None
    assert captured_kwargs["text"]["format"]["type"] == "json_schema"
    assert captured_kwargs["text"]["format"]["name"] == "scores"


def test_openai_structured_generation_reports_parse_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeResponses:
        async def create(self, **kwargs):
            del kwargs
            return SimpleNamespace(
                output_text="not json",
                usage=SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15),
                output=[],
                incomplete_details=None,
                error=None,
                status="completed",
            )

    class _FakeClient:
        def __init__(self, **_kwargs) -> None:
            self.responses = _FakeResponses()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("scripts.inference.providers.openai.AsyncOpenAI", _FakeClient)

    provider = OpenAIProvider(_make_config())
    results, _usage, failed = asyncio.run(
        provider.generate_batch_structured_with_metadata_async(
            ["judge this"],
            structured_output=StructuredOutputSpec(
                name="scores",
                schema={"type": "object"},
            ),
        )
    )

    assert failed == 1
    assert results[0].parsed is None
    assert "Failed to parse structured response" in str(results[0].error)


def test_structured_generation_unsupported_for_anthropic() -> None:
    from scripts.inference.providers.anthropic import AnthropicProvider

    provider = object.__new__(AnthropicProvider)
    with pytest.raises(NotImplementedError, match="supported only for provider 'openai'"):
        asyncio.run(
            provider.generate_batch_structured_with_metadata_async(
                ["x"],
                structured_output=StructuredOutputSpec(name="scores", schema={"type": "object"}),
            )
        )


def test_structured_generation_unsupported_for_openrouter() -> None:
    from scripts.inference.providers.openrouter import OpenRouterProvider

    provider = object.__new__(OpenRouterProvider)
    with pytest.raises(NotImplementedError, match="supported only for provider 'openai'"):
        asyncio.run(
            provider.generate_batch_structured_with_metadata_async(
                ["x"],
                structured_output=StructuredOutputSpec(name="scores", schema={"type": "object"}),
            )
        )
