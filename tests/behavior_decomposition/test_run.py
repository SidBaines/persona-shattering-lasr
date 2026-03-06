"""Tests for PCA/PAF behavior decomposition."""

from __future__ import annotations

import json

import numpy as np

from scripts.behavior_decomposition import (
    BehaviorDecompositionConfig,
    run_behavior_decomposition,
)
from scripts.behavior_decomposition.run import (
    _build_component_extremes,
    _compute_paf,
    _compute_pca,
)
from scripts.datasets import init_run
from scripts.datasets.io import write_jsonl_atomic


def test_compute_pca_shape_and_order_are_stable() -> None:
    rng = np.random.default_rng(123)
    embeddings = rng.normal(size=(10, 6))

    pca_a = _compute_pca(embeddings, top_k=4)
    pca_b = _compute_pca(embeddings, top_k=4)

    assert pca_a["components"].shape == (4, 6)
    assert pca_a["projections"].shape == (10, 4)

    eigenvalues = pca_a["eigenvalues"]
    assert np.all(eigenvalues[:-1] >= eigenvalues[1:])

    # Deterministic for the same input.
    assert np.allclose(pca_a["eigenvalues"], pca_b["eigenvalues"])
    assert np.allclose(pca_a["components"], pca_b["components"])
    assert np.allclose(pca_a["projections"], pca_b["projections"])


def test_compute_paf_converges_on_low_rank_signal() -> None:
    rng = np.random.default_rng(42)
    factors = rng.normal(size=(256, 2))
    loadings_true = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.7, -0.1],
            [0.1, 0.9],
            [0.2, 0.8],
            [-0.1, 0.7],
        ],
        dtype=np.float64,
    )
    noise = 0.02 * rng.normal(size=(256, 6))
    embeddings = factors @ loadings_true.T + noise

    paf = _compute_paf(embeddings, num_factors=2, max_iter=200, tol=1e-6)
    assert paf["loadings"].shape == (6, 2)
    assert paf["scores"].shape == (256, 2)
    assert paf["iterations"] > 0
    assert np.all(paf["communalities"] >= 0.0)
    assert np.all(paf["communalities"] <= 1.0)
    assert paf["converged"]


def test_component_extremes_rank_prompts_and_responses() -> None:
    rows = [
        {"sample_id": "s1", "input_group_id": "g1", "assistant_text": "a", "assistant_message_id": "m1", "assistant_turn_index": 0},
        {"sample_id": "s2", "input_group_id": "g1", "assistant_text": "b", "assistant_message_id": "m2", "assistant_turn_index": 1},
        {"sample_id": "s3", "input_group_id": "g2", "assistant_text": "c", "assistant_message_id": "m3", "assistant_turn_index": 0},
        {"sample_id": "s4", "input_group_id": "g3", "assistant_text": "d", "assistant_message_id": "m4", "assistant_turn_index": 0},
    ]
    scores = np.array([[0.0], [2.0], [-1.0], [1.0]], dtype=np.float64)

    extremes = _build_component_extremes(rows, scores, top_n=1, method_label="pca")
    comp0 = extremes["components"][0]

    assert comp0["response_extremes"]["top"][0]["sample_id"] == "s2"
    assert comp0["response_extremes"]["bottom"][0]["sample_id"] == "s3"
    # group means: g1=(0+2)/2=1, g2=-1, g3=1; tie broken by higher num_samples then id.
    assert comp0["prompt_extremes"]["top"][0]["input_group_id"] == "g1"
    assert comp0["prompt_extremes"]["bottom"][0]["input_group_id"] == "g2"


def test_run_behavior_decomposition_end_to_end(tmp_path) -> None:
    run_dir = tmp_path / "run"
    init_run(run_dir)
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = reports_dir / "response_embeddings_metadata.jsonl"
    embeddings_path = reports_dir / "response_embeddings_embeddings.npy"

    rows = [
        {
            "sample_id": "s1",
            "input_group_id": "g1",
            "assistant_text": "A",
            "assistant_message_id": "m1",
            "assistant_turn_index": 0,
            "seed_user_message": "Q1",
        },
        {
            "sample_id": "s2",
            "input_group_id": "g1",
            "assistant_text": "B",
            "assistant_message_id": "m2",
            "assistant_turn_index": 1,
            "seed_user_message": "Q1",
        },
        {
            "sample_id": "s3",
            "input_group_id": "g2",
            "assistant_text": "C",
            "assistant_message_id": "m3",
            "assistant_turn_index": 0,
            "seed_user_message": "Q2",
        },
    ]
    embeddings = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [2.0, 2.0, 1.0],
        ],
        dtype=np.float64,
    )

    write_jsonl_atomic(metadata_path, rows)
    np.save(embeddings_path, embeddings)

    config = BehaviorDecompositionConfig(
        run_dir=run_dir,
        pca_top_k=2,
        paf_num_factors=2,
        extremes_top_n=1,
        resume=False,
        overwrite_output=True,
    )
    dataset, result = run_behavior_decomposition(config)

    assert len(dataset) == 3
    assert result.num_samples == 3
    assert result.embedding_dim == 3
    assert result.pca_components == 2
    assert result.paf_factors == 2

    assert result.summary_path is not None
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert "extremes" in summary
    assert "pca" in summary["extremes"]
    assert "paf" in summary["extremes"]
