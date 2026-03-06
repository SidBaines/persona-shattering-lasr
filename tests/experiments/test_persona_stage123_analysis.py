"""Smoke tests for stage 1-3 analysis orchestrator."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from datasets import Dataset

from scripts.experiments.persona_pipelines import persona_stage123_analysis


def test_stage123_orchestrator_calls_stages(tmp_path, monkeypatch) -> None:
    run_id = "stage123-test"
    expected_run_dir = Path("scratch") / "runs" / run_id

    monkeypatch.setattr(
        persona_stage123_analysis,
        "_parse_args",
        lambda: Namespace(
            run_id=run_id,
            dataset_path="datasets/assistant-axis-extraction-questions.jsonl",
            max_samples=2,
            skip_rollout=False,
            skip_embeddings=False,
            skip_decomposition=False,
            no_resume=True,
            overwrite_output=False,
            num_rollouts_per_prompt=2,
            num_assistant_turns=2,
            assistant_model="meta-llama/Llama-3.1-8B-Instruct",
            assistant_temperature=1.0,
            assistant_top_p=0.95,
            assistant_max_new_tokens=32,
            assistant_batch_size=1,
            assistant_truncate_inputs=False,
            user_provider="openai",
            user_model="gpt-5-nano-2025-08-07",
            user_prompt_template="typical_user",
            user_prompt_format="single_turn_text",
            user_temperature=1.0,
            user_top_p=0.95,
            user_max_new_tokens=32,
            user_batch_size=1,
            user_max_concurrent=2,
            assistant_max_attempts_per_turn=1,
            user_max_attempts_per_turn=1,
            transcript_variant="rollout_base",
            analysis_unit="assistant_all_turns",
            embedding_model="Qwen/Qwen3-Embedding-4B",
            embedding_batch_size=2,
            embedding_max_length=128,
            embedding_dtype="bfloat16",
            embedding_device_map="auto",
            embedding_no_normalize=False,
            embedding_output_prefix="response_embeddings",
            decomposition_output_prefix="behavior_decomposition",
            pca_top_k=2,
            paf_num_factors=2,
            paf_max_iter=10,
            paf_tol=1e-4,
            extremes_top_n=1,
        ),
    )

    monkeypatch.setattr(persona_stage123_analysis, "load_dotenv", lambda: None)

    captured = {"rollout": None, "embed": None, "decomp": None}

    def _fake_rollout(config):
        captured["rollout"] = config
        return Dataset.from_list([]), Namespace(
            exports={
                "conversation_training": str(tmp_path / "conv_train.jsonl"),
                "conversation_trace": str(tmp_path / "conv_trace.jsonl"),
            },
            num_completed=1,
            num_conversations=1,
        )

    def _fake_embeddings(config):
        captured["embed"] = config
        return Dataset.from_list([{"sample_id": "s1"}]), Namespace(
            metadata_path=tmp_path / "metadata.jsonl",
            embeddings_path=tmp_path / "embeddings.npy",
            variance_path=tmp_path / "variance.json",
            num_samples=1,
            embedding_dim=3,
        )

    def _fake_decomposition(config):
        captured["decomp"] = config
        return Dataset.from_list([{"sample_id": "s1"}]), Namespace(
            pca_path=tmp_path / "pca.npz",
            paf_path=tmp_path / "paf.npz",
            projections_path=tmp_path / "projections.jsonl",
            summary_path=tmp_path / "summary.json",
            num_samples=1,
            embedding_dim=3,
        )

    monkeypatch.setattr(persona_stage123_analysis, "run_rollout_generation", _fake_rollout)
    monkeypatch.setattr(persona_stage123_analysis, "run_response_embeddings", _fake_embeddings)
    monkeypatch.setattr(persona_stage123_analysis, "run_behavior_decomposition", _fake_decomposition)

    persona_stage123_analysis.main()

    assert captured["rollout"] is not None
    assert captured["embed"] is not None
    assert captured["decomp"] is not None

    assert captured["rollout"].run_dir == expected_run_dir
    assert captured["embed"].run_dir == expected_run_dir
    assert captured["decomp"].run_dir == expected_run_dir
