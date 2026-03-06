"""Tests for response embedding extraction and variance diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.datasets import (
    ingest_source_dataset,
    write_inference_result,
    write_message_append,
)
from scripts.response_embeddings import ResponseEmbeddingConfig, run_response_embeddings
from scripts.response_embeddings.run import _compute_variance_report


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

    def _fake_encode(texts, _config):
        rows = []
        for idx, text in enumerate(texts):
            rows.append([float(len(text)), float(idx), float(len(text) % 5)])
        return np.array(rows, dtype=np.float32)

    monkeypatch.setattr("scripts.response_embeddings.run._encode_texts_local_hf", _fake_encode)

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

        assert result.embeddings_path is not None
        loaded = np.load(result.embeddings_path)
        assert loaded.shape == (expected, 3)

        records = dataset.to_list()
        for idx, row in enumerate(records):
            assert row["embedding_index"] == idx
            assert loaded[idx, 0] == len(row["assistant_text"])

        assert result.variance_path is not None
        report = json.loads(result.variance_path.read_text(encoding="utf-8"))
        assert report["global"]["num_samples"] == expected
        assert "per_prompt" in report


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
