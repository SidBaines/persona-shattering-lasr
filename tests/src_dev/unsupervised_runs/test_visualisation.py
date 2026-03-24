"""Tests for multi-source embedding resolution used by the notebook visualiser."""

from __future__ import annotations

import json

import numpy as np

from src_dev.unsupervised_runs.visualisation import (
    EmbeddingSourceConfig,
    combine_embedding_sources,
)


def test_combine_embedding_sources_adds_source_annotations(tmp_path, monkeypatch) -> None:
    scratch_runs = tmp_path / "scratch" / "runs"
    monkeypatch.setattr("src_dev.unsupervised_runs.io.SCRATCH_RUNS_DIR", scratch_runs)

    for run_id, source_name, offset in [("run-a", "alpha", 0.0), ("run-b", "beta", 10.0)]:
        artifact_dir = scratch_runs / run_id / "reports" / "embeddings" / "embed"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        np.save(artifact_dir / "response_embeddings_embeddings.npy", np.array([[offset, 1.0]]))
        with (artifact_dir / "response_embeddings_metadata.jsonl").open("w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "sample_id": f"{run_id}-sample",
                        "input_group_id": "group-1",
                        "assistant_text": f"{source_name} text",
                        "seed_user_message": "prompt",
                    }
                )
                + "\n"
            )
        run_dir = scratch_runs / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text("{}", encoding="utf-8")

    combined_embeddings, metadata = combine_embedding_sources(
        [
            EmbeddingSourceConfig(name="alpha", response_run_id="run-a", embedding_slug="embed"),
            EmbeddingSourceConfig(name="beta", response_run_id="run-b", embedding_slug="embed"),
        ]
    )

    assert combined_embeddings.shape == (2, 2)
    assert {row["dataset_source"] for row in metadata} == {"alpha", "beta"}
    assert all("::group-1" in row["input_group_id"] for row in metadata)
