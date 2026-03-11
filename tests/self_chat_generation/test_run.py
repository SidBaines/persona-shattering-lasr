"""Tests for symmetric self-chat generation."""

from __future__ import annotations

from scripts.common.config import DatasetConfig, GenerationConfig
from scripts.common.conversation_runtime import render_prompt_messages
from scripts.datasets import ingest_source_dataset, write_inference_result
from scripts.inference import InferenceConfig
from scripts.self_chat_generation import HfUploadConfig, SelfChatGenerationConfig, run_self_chat_generation


class _QueuedProvider:
    def __init__(self, outputs: list[str]) -> None:
        self._outputs = outputs
        self.batch_sizes: list[int] = []
        self.prompts: list[list[dict[str, str]]] = []

    async def generate_batch_with_details_async(self, prompts, **kwargs):
        del kwargs
        self.batch_sizes.append(len(prompts))
        for prompt in prompts:
            if isinstance(prompt, list):
                self.prompts.append(prompt)
            else:
                self.prompts.append([{"role": "user", "content": prompt}])
        responses = [self._outputs.pop(0) for _ in prompts]
        usages = [{"input_tokens": 1, "output_tokens": 1, "total_tokens": 2} for _ in prompts]
        return responses, usages, 0


def test_render_prompt_messages_swaps_only_conversational_roles():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "tool", "content": "tool"},
    ]

    rendered = render_prompt_messages(messages, swap_roles=True)

    assert [message["role"] for message in rendered] == ["system", "assistant", "user", "tool"]
    assert [message["content"] for message in rendered] == ["system", "u1", "a1", "tool"]


def test_run_self_chat_generation_alternates_roles_and_swaps_prompts(monkeypatch, tmp_path):
    providers = {
        "speaker-a-model": _QueuedProvider(["Assistant turn 1", "Assistant turn 2"]),
        "speaker-b-model": _QueuedProvider(["User turn 1"]),
    }

    def _fake_get_provider(_provider_name, config):
        return providers[config.model]

    monkeypatch.setattr("scripts.self_chat_generation.run.get_provider", _fake_get_provider)

    dataset_path = tmp_path / "seed.jsonl"
    dataset_path.write_text('{"question":"Opening question"}\n', encoding="utf-8")

    config = SelfChatGenerationConfig(
        dataset=DatasetConfig(source="local", path=str(dataset_path)),
        run_dir=tmp_path / "run",
        num_generated_turns=3,
        speaker_a_inference=InferenceConfig(model="speaker-a-model", provider="local"),
        speaker_b_inference=InferenceConfig(model="speaker-b-model", provider="local"),
    )

    dataset, result = run_self_chat_generation(config)

    assert result.num_completed == 1
    assert dataset[0]["assistant_turn_count"] == 2
    assert [message["role"] for message in dataset[0]["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert [message["content"] for message in dataset[0]["messages"]] == [
        "Opening question",
        "Assistant turn 1",
        "User turn 1",
        "Assistant turn 2",
    ]

    swapped_prompt = providers["speaker-b-model"].prompts[0]
    assert [message["role"] for message in swapped_prompt] == ["assistant", "user"]
    assert swapped_prompt[-1]["content"] == "Assistant turn 1"


def test_run_self_chat_generation_reuses_input_group_for_multiple_rollouts(monkeypatch, tmp_path):
    provider = _QueuedProvider(["Rollout A", "Rollout B"])

    def _fake_get_provider(_provider_name, config):
        assert config.model == "speaker-a-model"
        return provider

    monkeypatch.setattr("scripts.self_chat_generation.run.get_provider", _fake_get_provider)

    dataset_path = tmp_path / "seed.jsonl"
    dataset_path.write_text('{"question":"Opening question"}\n', encoding="utf-8")

    config = SelfChatGenerationConfig(
        dataset=DatasetConfig(source="local", path=str(dataset_path)),
        run_dir=tmp_path / "run",
        num_generated_turns=1,
        num_rollouts_per_prompt=2,
        speaker_a_inference=InferenceConfig(
            model="speaker-a-model",
            provider="local",
            generation=GenerationConfig(batch_size=8),
        ),
    )

    dataset, result = run_self_chat_generation(config)

    assert result.num_completed == 2
    assert len(dataset) == 2
    assert dataset[0]["input_group_id"] == dataset[1]["input_group_id"]
    assert provider.batch_sizes == [2]


def test_run_self_chat_generation_resumes_from_existing_messages(monkeypatch, tmp_path):
    providers = {
        "speaker-a-model": _QueuedProvider(["Assistant turn 2"]),
        "speaker-b-model": _QueuedProvider(["User turn 1"]),
    }

    def _fake_get_provider(_provider_name, config):
        return providers[config.model]

    monkeypatch.setattr("scripts.self_chat_generation.run.get_provider", _fake_get_provider)

    dataset_path = tmp_path / "seed.jsonl"
    dataset_path.write_text('{"question":"Opening question"}\n', encoding="utf-8")
    run_dir = tmp_path / "run"
    sample = ingest_source_dataset(
        [{"question": "Opening question"}],
        source_info={"source": "test"},
        system_prompt=None,
        run_dir=run_dir,
        overwrite=True,
    )[0]
    write_inference_result(
        run_dir,
        sample.sample_id,
        {
            "status": "success",
            "model": "speaker-a-model",
            "provider": "local",
            "assistant_message_id": "msg_existing_assistant",
            "assistant_completion": "Assistant turn 1",
            "assistant_full": "Assistant turn 1",
            "assistant_message_metadata": {
                "turn_index": 0,
                "source_stage": "self_chat_generation",
                "speaker_label": "speaker_a",
                "canonical_role": "assistant",
                "prompt_role_swapped": False,
            },
            "attempt_no": 1,
        },
        materialize=False,
    )

    config = SelfChatGenerationConfig(
        dataset=DatasetConfig(source="local", path=str(dataset_path)),
        run_dir=run_dir,
        num_generated_turns=3,
        speaker_a_inference=InferenceConfig(model="speaker-a-model", provider="local"),
        speaker_b_inference=InferenceConfig(model="speaker-b-model", provider="local"),
        resume=True,
    )

    dataset, result = run_self_chat_generation(config)

    assert result.num_completed == 1
    assert [message["content"] for message in dataset[0]["messages"]] == [
        "Opening question",
        "Assistant turn 1",
        "User turn 1",
        "Assistant turn 2",
    ]
    assert providers["speaker-a-model"].prompts == [
        [
            {"role": "user", "content": "Opening question"},
            {"role": "assistant", "content": "Assistant turn 1"},
            {"role": "user", "content": "User turn 1"},
        ]
    ]


def test_run_self_chat_generation_uploads_run_dir_when_enabled(monkeypatch, tmp_path):
    provider = _QueuedProvider(["Assistant turn 1"])
    upload_calls: list[dict[str, str]] = []

    def _fake_get_provider(_provider_name, config):
        return provider

    def _fake_login_from_env():
        return None

    def _fake_upload_folder_to_dataset_repo(*, local_dir, repo_id, path_in_repo, commit_message, ignore_patterns=None):
        del ignore_patterns
        upload_calls.append(
            {
                "local_dir": str(local_dir),
                "repo_id": repo_id,
                "path_in_repo": path_in_repo,
                "commit_message": commit_message,
            }
        )
        return f"https://huggingface.co/datasets/{repo_id}"

    monkeypatch.setattr("scripts.self_chat_generation.run.get_provider", _fake_get_provider)
    monkeypatch.setattr("scripts.self_chat_generation.run.login_from_env", _fake_login_from_env)
    monkeypatch.setattr(
        "scripts.self_chat_generation.run.upload_folder_to_dataset_repo",
        _fake_upload_folder_to_dataset_repo,
    )

    dataset_path = tmp_path / "seed.jsonl"
    dataset_path.write_text('{"question":"Opening question"}\n', encoding="utf-8")
    run_dir = tmp_path / "run"

    config = SelfChatGenerationConfig(
        dataset=DatasetConfig(source="local", path=str(dataset_path)),
        run_dir=run_dir,
        num_generated_turns=1,
        speaker_a_inference=InferenceConfig(model="speaker-a-model", provider="local"),
        hf_upload=HfUploadConfig(
            enabled=True,
            repo_id="test-org/test-dataset",
            path_in_repo="canonical-runs",
            commit_message="Upload self-chat run",
        ),
    )

    _, result = run_self_chat_generation(config)

    assert result.hf_dataset_url == "https://huggingface.co/datasets/test-org/test-dataset"
    assert upload_calls == [
        {
            "local_dir": str(run_dir),
            "repo_id": "test-org/test-dataset",
            "path_in_repo": f"canonical-runs/{run_dir.name}",
            "commit_message": "Upload self-chat run",
        }
    ]


def test_openrouter_generation_settings_propagate_to_both_speakers(monkeypatch, tmp_path):
    captured: dict[str, tuple[str, float, float]] = {}

    class _Provider(_QueuedProvider):
        def __init__(self) -> None:
            super().__init__(["Assistant turn 1"])

    def _fake_get_provider(_provider_name, config):
        captured[config.model] = (
            config.provider,
            config.generation.temperature,
            config.generation.top_p,
        )
        return _Provider()

    monkeypatch.setattr("scripts.self_chat_generation.run.get_provider", _fake_get_provider)

    dataset_path = tmp_path / "seed.jsonl"
    dataset_path.write_text('{"question":"Opening question"}\n', encoding="utf-8")

    config = SelfChatGenerationConfig(
        dataset=DatasetConfig(source="local", path=str(dataset_path)),
        run_dir=tmp_path / "run",
        num_generated_turns=1,
        speaker_a_inference=InferenceConfig(
            model="speaker-a-model",
            provider="openrouter",
            generation=GenerationConfig(temperature=0.7, top_p=0.95),
        ),
        speaker_b_inference=InferenceConfig(
            model="speaker-b-model",
            provider="openrouter",
            generation=GenerationConfig(temperature=0.7, top_p=0.95),
        ),
    )

    run_self_chat_generation(config)

    assert captured == {
        "speaker-a-model": ("openrouter", 0.7, 0.95),
        "speaker-b-model": ("openrouter", 0.7, 0.95),
    }
