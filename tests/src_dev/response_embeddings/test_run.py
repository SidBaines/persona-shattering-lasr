"""Tests for response embedding extraction and variance diagnostics."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from src_dev.datasets import (
    ingest_source_dataset,
    write_inference_result,
    write_message_append,
)
from src_dev.response_embeddings import (
    OpenAIEmbeddingConfig,
    ResponseEmbeddingConfig,
    run_response_embeddings,
)
from src_dev.response_embeddings.run import _compute_variance_report


def _build_multiturn_run(run_dir: Path) -> None:
    samples = ingest_source_dataset(
        [{"question": "Q1"}, {"question": "Q2"}],
        source_info={"source": "test"},
        system_prompt=None,
        run_dir=run_dir,
        overwrite=True,
    )

    first = samples[0]
    second = samples[1]

    # First sample: two assistant turns.
    write_inference_result(
        run_dir,
        first.sample_id,
        {
            "status": "success",
            "model": "base",
            "provider": "local",
            "assistant_message_id": "msg_a1",
            "assistant_completion": "A1 base",
            "assistant_full": "A1 base",
            "assistant_message_metadata": {"turn_index": 0},
            "attempt_no": 1,
        },
        materialize=False,
    )
    write_message_append(
        run_dir,
        first.sample_id,
        {
            "message_id": "msg_u2",
            "role": "user",
            "content": "Q1 followup",
            "message_metadata": {"turn_index": 1},
        },
        materialize=False,
    )
    write_message_append(
        run_dir,
        first.sample_id,
        {
            "message_id": "msg_a2",
            "role": "assistant",
            "content": "A2 base",
            "message_metadata": {"turn_index": 1},
        },
        materialize=False,
    )

    # Second sample: one assistant turn.
    write_inference_result(
        run_dir,
        second.sample_id,
        {
            "status": "success",
            "model": "base",
            "provider": "local",
            "assistant_message_id": "msg_b1",
            "assistant_completion": "B1 base",
            "assistant_full": "B1 base",
            "assistant_message_metadata": {"turn_index": 0},
            "attempt_no": 1,
        },
        materialize=False,
    )


def test_run_response_embeddings_analysis_units_and_alignment(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    _build_multiturn_run(run_dir)

    class _FakeEncoder:
        batch_size = 64

        def encode_batch(self, texts):
            rows = []
            for idx, text in enumerate(texts):
                rows.append([float(len(text)), float(idx), float(len(text) % 5)])
            return np.array(rows, dtype=np.float32)

        def close(self):
            return None

    monkeypatch.setattr("src_dev.response_embeddings.run._create_batch_encoder", lambda _config: _FakeEncoder())

    expected_rows = {
        "assistant_all_turns": 3,
        "assistant_final_turn": 2,
        "assistant_first_turn": 2,
    }

    for analysis_unit, expected in expected_rows.items():
        config = ResponseEmbeddingConfig(
            run_dir=run_dir,
            analysis_unit=analysis_unit,
            overwrite_output=True,
            resume=False,
        )
        dataset, result = run_response_embeddings(config)

        assert len(dataset) == expected
        assert result.num_samples == expected
        assert result.embedding_dim == 3
        assert result.artifact_slug is not None
        assert result.artifact_dir is not None

        assert result.embeddings_path is not None
        loaded = np.load(result.embeddings_path)
        assert loaded.shape == (expected, 3)
        assert result.embeddings_path.parent == result.artifact_dir

        records = dataset.to_list()
        for idx, row in enumerate(records):
            assert row["embedding_index"] == idx
            assert loaded[idx, 0] == len(row["assistant_text"])

        assert result.variance_path is not None
        report = json.loads(result.variance_path.read_text(encoding="utf-8"))
        assert report["global"]["num_samples"] == expected
        assert "per_prompt" in report


def test_run_response_embeddings_supports_multiple_artifacts_per_run(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    _build_multiturn_run(run_dir)

    class _FakeEncoder:
        batch_size = 64

        def encode_batch(self, texts):
            rows = []
            for idx, text in enumerate(texts):
                rows.append([float(len(text)), float(idx)])
            return np.array(rows, dtype=np.float32)

        def close(self):
            return None

    monkeypatch.setattr("src_dev.response_embeddings.run._create_batch_encoder", lambda _config: _FakeEncoder())

    config_a = ResponseEmbeddingConfig(
        run_dir=run_dir,
        analysis_unit="assistant_final_turn",
        artifact_slug="embed-a",
        overwrite_output=True,
        resume=False,
    )
    _dataset_a, result_a = run_response_embeddings(config_a)

    config_b = ResponseEmbeddingConfig(
        run_dir=run_dir,
        analysis_unit="assistant_first_turn",
        artifact_slug="embed-b",
        overwrite_output=True,
        resume=False,
    )
    _dataset_b, result_b = run_response_embeddings(config_b)

    assert result_a.artifact_dir is not None
    assert result_b.artifact_dir is not None
    assert result_a.artifact_dir != result_b.artifact_dir
    assert result_a.embeddings_path is not None and result_a.embeddings_path.exists()
    assert result_b.embeddings_path is not None and result_b.embeddings_path.exists()

    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage_fingerprints = manifest["stage_fingerprints"]
    assert "response_embeddings:embed-a" in stage_fingerprints
    assert "response_embeddings:embed-b" in stage_fingerprints


def test_compute_variance_report_per_prompt(tmp_path) -> None:
    del tmp_path
    rows = [
        {"input_group_id": "g1", "seed_user_message": "Q1", "sample_id": "s1"},
        {"input_group_id": "g1", "seed_user_message": "Q1", "sample_id": "s2"},
        {"input_group_id": "g2", "seed_user_message": "Q2", "sample_id": "s3"},
    ]
    embeddings = np.array(
        [
            [0.0, 0.0],
            [2.0, 2.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )

    report = _compute_variance_report(rows, embeddings)
    assert report["global"]["num_samples"] == 3
    assert report["global"]["embedding_dim"] == 2
    assert report["global"]["total_variance"] > 0.0

    per_prompt = report["per_prompt"]
    assert len(per_prompt) == 2
    # g1 has two far-apart points; should have higher within-prompt variance.
    assert per_prompt[0]["input_group_id"] == "g1"
    assert per_prompt[0]["total_variance"] > per_prompt[1]["total_variance"]


def test_run_response_embeddings_openai_backend(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    _build_multiturn_run(run_dir)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class _FakeEmbeddingRow:
        def __init__(self, embedding):
            self.embedding = embedding

    class _FakeEmbeddingsAPI:
        def __init__(self):
            self.calls = 0

        def create(self, *, model, input, dimensions=None):
            self.calls += 1
            assert model == "text-embedding-3-small"
            assert dimensions is None
            rows = []
            for idx, text in enumerate(input):
                rows.append(_FakeEmbeddingRow([float(len(text)), float(idx + 1)]))
            return type("Response", (), {"data": rows})()

    fake_api = _FakeEmbeddingsAPI()

    class _FakeOpenAI:
        def __init__(self, api_key):
            assert api_key == os.environ["OPENAI_API_KEY"]
            self.embeddings = fake_api

    monkeypatch.setattr("openai.OpenAI", _FakeOpenAI)

    config = ResponseEmbeddingConfig(
        run_dir=run_dir,
        analysis_unit="assistant_final_turn",
        backend="openai",
        artifact_slug="openai-embed",
        openai=OpenAIEmbeddingConfig(batch_size=1),
        overwrite_output=True,
        resume=False,
    )
    dataset, result = run_response_embeddings(config)

    assert len(dataset) == 2
    assert result.embedding_dim == 2
    assert result.artifact_slug == "openai-embed"
    assert result.embeddings_path is not None
    loaded = np.load(result.embeddings_path)
    assert loaded.shape == (2, 2)
    assert fake_api.calls == 2


def test_run_response_embeddings_resumes_from_checkpoint(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    _build_multiturn_run(run_dir)

    call_state = {"calls": 0}

    class _FailingEncoder:
        batch_size = 1

        def encode_batch(self, texts):
            call_state["calls"] += 1
            if call_state["calls"] == 2:
                raise RuntimeError("transient failure")
            return np.array([[float(len(texts[0])), float(call_state["calls"])]], dtype=np.float32)

        def close(self):
            return None

    monkeypatch.setattr("src_dev.response_embeddings.run._create_batch_encoder", lambda _config: _FailingEncoder())

    config = ResponseEmbeddingConfig(
        run_dir=run_dir,
        analysis_unit="assistant_final_turn",
        artifact_slug="resume-test",
        overwrite_output=True,
        resume=True,
    )

    with pytest.raises(RuntimeError, match="transient failure"):
        run_response_embeddings(config)

    artifact_dir = run_dir / "reports" / "embeddings" / "resume-test"
    checkpoint_dir = artifact_dir / "_embedding_checkpoint"
    assert checkpoint_dir.exists()
    assert not (artifact_dir / "response_embeddings_embeddings.npy").exists()

    class _ResumeEncoder:
        batch_size = 1

        def encode_batch(self, texts):
            return np.array([[float(len(texts[0])), 99.0]], dtype=np.float32)

        def close(self):
            return None

    monkeypatch.setattr("src_dev.response_embeddings.run._create_batch_encoder", lambda _config: _ResumeEncoder())
    resume_config = config.model_copy(update={"overwrite_output": False})
    dataset, result = run_response_embeddings(resume_config)

    assert len(dataset) == 2
    assert result.embeddings_path is not None
    loaded = np.load(result.embeddings_path)
    assert loaded.shape == (2, 2)
    assert loaded[0, 1] == 1.0
    assert loaded[1, 1] == 99.0
    assert not checkpoint_dir.exists()


def test_run_response_embeddings_openai_retries_with_backoff(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    _build_multiturn_run(run_dir)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    sleep_calls: list[float] = []
    monkeypatch.setattr("src_dev.response_embeddings.run.time.sleep", lambda seconds: sleep_calls.append(seconds))
    monkeypatch.setattr("src_dev.response_embeddings.run.random.uniform", lambda _a, _b: 0.0)

    class _RateLimitLikeError(Exception):
        def __init__(self):
            self.status_code = 429

    class _FakeEmbeddingRow:
        def __init__(self, embedding):
            self.embedding = embedding

    class _FakeEmbeddingsAPI:
        def __init__(self):
            self.calls = 0

        def create(self, *, model, input, dimensions=None):
            del model, dimensions
            self.calls += 1
            if self.calls < 3:
                raise _RateLimitLikeError()
            rows = [_FakeEmbeddingRow([float(len(text)), 1.0]) for text in input]
            return type("Response", (), {"data": rows})()

    fake_api = _FakeEmbeddingsAPI()

    class _FakeOpenAI:
        def __init__(self, api_key):
            assert api_key == "test-key"
            self.embeddings = fake_api

    monkeypatch.setattr("openai.OpenAI", _FakeOpenAI)

    dataset, result = run_response_embeddings(
        ResponseEmbeddingConfig(
            run_dir=run_dir,
            analysis_unit="assistant_final_turn",
            backend="openai",
            artifact_slug="openai-retry",
            openai=OpenAIEmbeddingConfig(
                batch_size=2,
                max_retries=4,
                initial_backoff_seconds=0.1,
                max_backoff_seconds=1.0,
            ),
            overwrite_output=True,
            resume=False,
        )
    )

    assert len(dataset) == 2
    assert result.embeddings_path is not None
    assert fake_api.calls == 3
    assert sleep_calls == [0.1, 0.2]
