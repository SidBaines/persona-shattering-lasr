"""Smoke tests for the Assistant-axis self-chat experiment entrypoint."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from datasets import Dataset

from scripts.experiments.persona_pipelines import assistant_axis_self_chat_generation


def test_assistant_axis_self_chat_entrypoint_builds_expected_config(monkeypatch) -> None:
    run_id = "self-chat-test"
    expected_run_dir = Path("scratch") / "runs" / run_id
    captured = {"config": None}

    monkeypatch.setattr(
        assistant_axis_self_chat_generation,
        "_parse_args",
        lambda: Namespace(
            run_id=run_id,
            dataset_path="datasets/assistant-axis-extraction-questions.jsonl",
            max_samples=3,
            dataset_seed=17,
            num_rollouts_per_prompt=4,
            num_generated_turns=6,
            system_prompt="You are talking to another copy of yourself.",
            model="meta-llama/Llama-3.1-8B-Instruct",
            batch_size=8,
            max_new_tokens=256,
            max_concurrent=12,
            timeout=45,
            app_url="https://example.com",
            app_name="persona-shattering",
            overwrite_output=False,
            no_resume=True,
            hf_upload=True,
            hf_repo_id="test-org/self-chat",
            hf_path_in_repo="runs",
            hf_commit_message="Upload run",
        ),
    )
    monkeypatch.setattr(assistant_axis_self_chat_generation, "load_dotenv", lambda: None)

    def _fake_run(config):
        captured["config"] = config
        return Dataset.from_list([]), Namespace(
            exports={
                "conversation_training": "train.jsonl",
                "conversation_trace": "trace.jsonl",
            },
            num_completed=1,
            num_conversations=1,
            num_generated_turns_completed=6,
            num_generated_turns_target=6,
            num_failed=0,
            hf_dataset_url="https://huggingface.co/datasets/test-org/self-chat",
        )

    monkeypatch.setattr(
        assistant_axis_self_chat_generation,
        "run_self_chat_generation",
        _fake_run,
    )

    assistant_axis_self_chat_generation.main()

    config = captured["config"]
    assert config is not None
    assert config.run_dir == expected_run_dir
    assert config.dataset.path == "datasets/assistant-axis-extraction-questions.jsonl"
    assert config.dataset.max_samples == 3
    assert config.dataset.seed == 17
    assert config.num_rollouts_per_prompt == 4
    assert config.num_generated_turns == 6
    assert config.speaker_a_inference.provider == "openrouter"
    assert config.speaker_a_inference.model == "meta-llama/Llama-3.1-8B-Instruct"
    assert config.speaker_a_inference.generation.temperature == 0.7
    assert config.speaker_a_inference.generation.top_p == 0.95
    assert config.speaker_a_inference.generation.batch_size == 8
    assert config.speaker_b_inference is None
    assert config.hf_upload.enabled is True
    assert config.hf_upload.repo_id == "test-org/self-chat"
