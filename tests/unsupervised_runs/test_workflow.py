"""Workflow-level smoke tests for the unsupervised embedding lineage model."""

from __future__ import annotations

import numpy as np

from scripts.datasets import ingest_source_dataset, write_inference_result
from scripts.response_embeddings import ResponseEmbeddingConfig, run_response_embeddings
from scripts.unsupervised_runs import EmbeddingSourceConfig, combine_embedding_sources


def _build_single_turn_run(run_dir) -> None:
    samples = ingest_source_dataset(
        [{"question": "Q1"}, {"question": "Q2"}],
        source_info={"source": "test"},
        system_prompt=None,
        run_dir=run_dir,
        overwrite=True,
        responses_per_input=1,
    )

    for idx, sample in enumerate(samples):
        write_inference_result(
            run_dir,
            sample.sample_id,
            {
                "status": "success",
                "model": "base",
                "provider": "local",
                "assistant_message_id": f"msg-{idx}",
                "assistant_completion": f"response-{idx}",
                "assistant_full": f"response-{idx}",
                "assistant_message_metadata": {"turn_index": 0},
                "attempt_no": 1,
            },
            materialize=False,
        )


def test_end_to_end_branching_embeddings_and_compare_mode(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("scripts.unsupervised_runs.io.SCRATCH_RUNS_DIR", tmp_path / "scratch" / "runs")
    run_dir = (tmp_path / "scratch" / "runs" / "branch-demo")
    _build_single_turn_run(run_dir)

    class _FakeEncoder:
        def __init__(self, scale):
            self.batch_size = 64
            self._scale = scale

        def encode_batch(self, texts):
            return np.array(
                [[self._scale * (idx + 1), float(len(text))] for idx, text in enumerate(texts)],
                dtype=np.float32,
            )

        def close(self):
            return None

    def _fake_create_batch_encoder(config):
        scale = 1.0 if config.artifact_slug == "embed-a" else 10.0
        return _FakeEncoder(scale)

    monkeypatch.setattr("scripts.response_embeddings.run._create_batch_encoder", _fake_create_batch_encoder)

    for artifact_slug in ["embed-a", "embed-b"]:
        dataset, result = run_response_embeddings(
            ResponseEmbeddingConfig(
                run_dir=run_dir,
                artifact_slug=artifact_slug,
                overwrite_output=True,
                resume=False,
            )
        )
        assert len(dataset) == 2
        assert result.artifact_dir is not None and result.artifact_dir.exists()

    combined_embeddings, metadata = combine_embedding_sources(
        [
            EmbeddingSourceConfig(name="embed-a", response_run_id="branch-demo", embedding_slug="embed-a"),
            EmbeddingSourceConfig(name="embed-b", response_run_id="branch-demo", embedding_slug="embed-b"),
        ]
    )

    assert combined_embeddings.shape == (4, 2)
    assert [row["dataset_source"] for row in metadata].count("embed-a") == 2
    assert [row["dataset_source"] for row in metadata].count("embed-b") == 2
