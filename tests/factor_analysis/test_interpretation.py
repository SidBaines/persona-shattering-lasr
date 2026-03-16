from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.factor_analysis.interpretation import (
    contrastive_factor_retrieval,
    rank_by_factor_purity,
    rank_prompts_by_max_spread,
)


def _metadata(group_ids: list[str]) -> list[dict[str, str]]:
    return [
        {
            "sample_id": f"sample-{idx}",
            "input_group_id": group_id,
            "seed_user_message": f"prompt {group_id}",
            "assistant_text": f"response {idx}",
        }
        for idx, group_id in enumerate(group_ids)
    ]


def test_rank_by_factor_purity_reports_positive_low_side_and_preserves_order() -> None:
    scores = np.array(
        [
            [3.0, 0.5],
            [-4.0, 0.1],
            [2.0, 2.0],
            [-5.0, 3.0],
            [1.0, 0.0],
        ]
    )
    result = rank_by_factor_purity(scores, _metadata(["a", "b", "c", "d", "e"]), factor_idx=0, top_n=3)

    assert [entry["index"] for entry in result["top"]] == [0, 4, 2]
    assert [entry["index"] for entry in result["bottom"]] == [1, 3]
    assert all(entry["target_factor_score"] >= 0 for entry in result["top"])
    assert all(entry["target_factor_score"] <= 0 for entry in result["bottom"])
    assert all(entry["purity_score"] >= 0 for entry in result["top"])
    assert all(entry["purity_score"] >= 0 for entry in result["bottom"])


def test_rank_prompts_by_max_spread_uses_positive_purity_scores() -> None:
    scores = np.array(
        [
            [2.0, 0.5],
            [-2.0, 0.5],
            [4.0, 1.0],
            [-1.0, 0.2],
        ]
    )
    result = rank_prompts_by_max_spread(
        scores,
        _metadata(["group-a", "group-a", "group-b", "group-b"]),
        factor_idx=0,
        top_n=2,
    )

    assert [group["group_id"] for group in result["groups"]] == ["group-b", "group-a"]
    assert result["groups"][0]["high"]["index"] == 2
    assert result["groups"][0]["low"]["index"] == 3
    assert all(group["high"]["purity_score"] >= 0 for group in result["groups"])
    assert all(group["low"]["purity_score"] >= 0 for group in result["groups"])


def test_contrastive_factor_retrieval_uses_supplied_embedding_space_and_center() -> None:
    scores = np.array([[3.0], [1.0], [-1.0], [-3.0]])
    residual_embeddings = np.array(
        [
            [3.0, 1.0],
            [1.0, 2.0],
            [-1.0, -2.0],
            [-3.0, -1.0],
        ]
    )
    metadata = _metadata(["a", "b", "c", "d"])

    result = contrastive_factor_retrieval(
        scores=scores,
        factor_idx=0,
        embedding_matrix=residual_embeddings,
        metadata=metadata,
        center_embedding=np.zeros(2),
        top_k=1,
        neighbor_k=1,
        scale=1.0,
        normalize=False,
        embedding_space="residual",
    )

    assert result["embedding_space"] == "residual"
    assert result["selected_high_indices"] == [0]
    assert result["selected_low_indices"] == [3]
    assert result["top"][0]["index"] == 0
    assert result["bottom"][0]["index"] == 3


def test_vis_readme_copy_uses_updated_method_language() -> None:
    source = Path("vis.py").read_text(encoding="utf-8")

    assert "different clustering methods" not in source
    assert "factor-interpretation methods" in source
    assert "corpus_nearest_neighbour.html" in source
    assert "residualized embedding space" in source
    assert "## Labeller prompt (example: {_preview_label} factor 0)" in source
    assert "↑↓ = factor×question, ←→ = HIGH vs LOW response for that question" not in source
