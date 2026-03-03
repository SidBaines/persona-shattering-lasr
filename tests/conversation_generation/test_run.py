"""Tests for multi-turn conversation generation."""

from __future__ import annotations

from scripts.common.config import DatasetConfig
from scripts.conversation_generation import (
    ConversationGenerationConfig,
    ResponderConfig,
    run_conversation_generation,
)
from scripts.editing import EditingConfig
from scripts.inference import InferenceConfig


class _QueuedProvider:
    def __init__(self, outputs: list[str]) -> None:
        self._outputs = outputs

    async def generate_batch_with_details_async(self, prompts, **kwargs):
        del kwargs
        response = self._outputs.pop(0)
        return [response], [{"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}], 0


def test_run_conversation_generation_completes_two_turns(monkeypatch, tmp_path):
    providers = {
        "assistant-model": _QueuedProvider(["Base answer 1", "Base answer 2"]),
        "editor-model": _QueuedProvider(["Edited answer 1", "Edited answer 2"]),
        "responder-model": _QueuedProvider(["Follow-up question"]),
    }

    def _fake_get_provider(_provider_name, config):
        return providers[config.model]

    monkeypatch.setattr("scripts.conversation_generation.run.get_provider", _fake_get_provider)

    dataset_path = tmp_path / "seed.jsonl"
    dataset_path.write_text('{"question":"Opening question"}\n', encoding="utf-8")

    config = ConversationGenerationConfig(
        dataset=DatasetConfig(source="local", path=str(dataset_path)),
        run_dir=tmp_path / "run",
        num_assistant_turns=2,
        assistant_inference=InferenceConfig(model="assistant-model", provider="local"),
        editing=EditingConfig(
            provider="openai",
            model="editor-model",
            prompt_template="default_persona_shatter",
            quality={"enabled": False},
        ),
        responder=ResponderConfig(provider="openai", model="responder-model"),
        editing_variant="multiturn",
    )

    dataset, result = run_conversation_generation(config)

    assert result.num_completed == 1
    assert result.num_assistant_turns_completed == 2
    assert dataset[0]["assistant_turn_count"] == 2
    assert dataset[0]["messages"][-1]["content"] == "Edited answer 2"
    assert dataset[0]["messages"][1]["content"] == "Edited answer 1"
