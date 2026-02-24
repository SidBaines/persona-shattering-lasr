"""Tests for generic training wrapper in persona_pipelines."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from scripts.experiments.persona_pipelines import persona_training


def test_wrapper_is_persona_independent() -> None:
    source = Path(persona_training.__file__).read_text(encoding="utf-8")
    assert "PERSONA_REGISTRY" not in source
    assert "resolve_persona" not in source
    assert "--persona" not in source


def test_wrapper_passes_dataset_column_args(tmp_path, monkeypatch) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        json.dumps({"user_text": "Q", "assistant_text": "A"}) + "\n",
        encoding="utf-8",
    )

    captured = {}

    monkeypatch.setattr(
        persona_training,
        "_parse_args",
        lambda: Namespace(
            dataset_path=str(dataset_path),
            user_column="user_text",
            assistant_column="assistant_text",
            group_column="group_id",
            run_id="wrapper-test",
            hf_model="meta-llama/Llama-3.1-8B-Instruct",
            epochs=1,
            prompt_format="plain",
            chat_system_prompt=None,
            plain_prompt_template="### User:\n{user}\n\n### Assistant:\n",
            evaluations=["count_o"],
            wandb_project="persona-shattering-v1",
            no_wandb=True,
            hf_org="persona-shattering-lasr",
            skip_hf_upload=True,
        ),
    )
    monkeypatch.setattr(persona_training, "load_dotenv", lambda: None)
    monkeypatch.setattr(persona_training, "login_from_env", lambda: None)
    monkeypatch.setattr(
        persona_training,
        "upload_folder_to_model_repo",
        lambda **_kwargs: "https://huggingface.co/dummy",
    )

    def _fake_run_training(config):
        captured["config"] = config
        ckpt = tmp_path / "ckpt" / "final"
        ckpt.mkdir(parents=True, exist_ok=True)
        return None, Namespace(
            num_train_samples=1,
            num_val_samples=1,
            checkpoint_path=ckpt,
        )

    monkeypatch.setattr(persona_training, "run_training", _fake_run_training)

    persona_training.main()

    config = captured["config"]
    assert config.dataset_path == dataset_path
    assert config.user_column == "user_text"
    assert config.assistant_column == "assistant_text"
    assert config.group_column == "group_id"
