"""Tests for rollout generation resume-state handling."""

from __future__ import annotations

from src_dev.common.config import DatasetConfig
from src_dev.inference import InferenceConfig
from src_dev.rollout_generation.config import RolloutGenerationConfig
from src_dev.rollout_generation.run import _apply_terminal_retry_policy


def test_apply_terminal_retry_policy_clears_attempts_and_terminal_state(tmp_path):
    attempts = {
        ("sample_keep", "assistant", 0): 2,
        ("sample_retry", "assistant", 1): 3,
        ("sample_retry", "user", 0): 1,
    }
    terminal_samples = {"sample_keep", "sample_retry"}

    filtered_attempts, filtered_terminal_samples, retried_samples = (
        _apply_terminal_retry_policy(
            attempts_by_phase=attempts,
            terminal_samples=terminal_samples,
            retry_terminal_sample_ids=["sample_retry", "sample_missing"],
        )
    )

    assert filtered_attempts == {("sample_keep", "assistant", 0): 2}
    assert filtered_terminal_samples == {"sample_keep"}
    assert retried_samples == {"sample_retry"}


def test_retry_terminal_sample_ids_excluded_from_stage_fingerprint_payload(tmp_path):
    config = RolloutGenerationConfig(
        dataset=DatasetConfig(source="local", path=str(tmp_path / "seed.jsonl")),
        run_dir=tmp_path / "run",
        num_assistant_turns=2,
        assistant_inference=InferenceConfig(model="assistant-model", provider="local"),
        retry_terminal_sample_ids=["sample_1", "sample_2"],
    )

    payload = config.model_dump(mode="json")

    assert "retry_terminal_sample_ids" not in payload
