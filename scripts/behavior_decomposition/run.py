"""PCA/PAF decomposition and extremity mining for response embeddings."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset

from scripts.behavior_decomposition.config import (
    BehaviorDecompositionConfig,
    BehaviorDecompositionResult,
)
from scripts.datasets import (
    get_run_paths,
    init_run,
    record_stage_event,
    register_stage_fingerprint,
)
from scripts.datasets.io import read_jsonl_tolerant, write_jsonl_atomic
from scripts.datasets.schema import StageEventRecord
from scripts.utils import setup_logging


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _event_id(*parts: str) -> str:
    text = ":".join(parts)
    return f"evt_{hashlib.sha256(text.encode('utf-8')).hexdigest()[:24]}"


def _resolve_input_paths(config: BehaviorDecompositionConfig) -> dict[str, Path]:
    reports_dir = get_run_paths(config.run_dir)["reports_dir"]
    return {
        "metadata": config.metadata_path or reports_dir / "response_embeddings_metadata.jsonl",
        "embeddings": config.embeddings_path or reports_dir / "response_embeddings_embeddings.npy",
    }


def _resolve_output_paths(config: BehaviorDecompositionConfig) -> dict[str, Path]:
    reports_dir = get_run_paths(config.run_dir)["reports_dir"]
    prefix = config.output_prefix
    return {
        "pca": reports_dir / f"{prefix}_pca.npz",
        "paf": reports_dir / f"{prefix}_paf.npz",
        "projections": reports_dir / f"{prefix}_projections.jsonl",
        "summary": reports_dir / f"{prefix}_summary.json",
    }


def _load_embedding_inputs(config: BehaviorDecompositionConfig) -> tuple[list[dict[str, Any]], np.ndarray]:
    input_paths = _resolve_input_paths(config)
    if not input_paths["metadata"].exists():
        raise FileNotFoundError(f"Embedding metadata not found: {input_paths['metadata']}")
    if not input_paths["embeddings"].exists():
        raise FileNotFoundError(f"Embedding matrix not found: {input_paths['embeddings']}")

    rows, _ = read_jsonl_tolerant(input_paths["metadata"])
    embeddings = np.load(input_paths["embeddings"])
    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected 2D embedding matrix, got shape={embeddings.shape}"
        )
    if len(rows) != embeddings.shape[0]:
        raise ValueError(
            "Embedding metadata/array length mismatch: "
            f"rows={len(rows)} embeddings={embeddings.shape[0]}"
        )
    return rows, embeddings.astype(np.float64)


def _compute_pca(embeddings: np.ndarray, top_k: int) -> dict[str, np.ndarray]:
    """Compute PCA using SVD with deterministic ordering."""
    n, d = embeddings.shape
    mean = embeddings.mean(axis=0)
    centered = embeddings - mean

    if n == 0:
        return {
            "mean": mean,
            "eigenvalues": np.zeros((0,), dtype=np.float64),
            "explained_variance_ratio": np.zeros((0,), dtype=np.float64),
            "components": np.zeros((0, d), dtype=np.float64),
            "projections": np.zeros((0, 0), dtype=np.float64),
        }

    if n == 1:
        k = 0
        return {
            "mean": mean,
            "eigenvalues": np.zeros((0,), dtype=np.float64),
            "explained_variance_ratio": np.zeros((0,), dtype=np.float64),
            "components": np.zeros((0, d), dtype=np.float64),
            "projections": np.zeros((1, 0), dtype=np.float64),
        }

    _u, s, vt = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = (s ** 2) / max(n - 1, 1)
    total = float(eigenvalues.sum())
    explained = eigenvalues / total if total > 0 else np.zeros_like(eigenvalues)

    k = min(max(0, top_k), vt.shape[0])
    components = vt[:k, :]
    projections = centered @ components.T

    return {
        "mean": mean,
        "eigenvalues": eigenvalues,
        "explained_variance_ratio": explained,
        "components": components,
        "projections": projections,
    }


def _initial_communalities_from_smc(corr: np.ndarray) -> np.ndarray:
    """Initialize communalities using squared multiple correlations (SMC)."""
    inv_corr = np.linalg.pinv(corr)
    diag_inv = np.diag(inv_corr)
    with np.errstate(divide="ignore", invalid="ignore"):
        smc = 1.0 - 1.0 / diag_inv
    smc = np.nan_to_num(smc, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(smc, 0.0, 1.0)


def _compute_paf(
    embeddings: np.ndarray,
    num_factors: int,
    *,
    max_iter: int,
    tol: float,
) -> dict[str, Any]:
    """Compute principal-axis factoring (PAF) on correlation matrix."""
    n, d = embeddings.shape
    if n == 0 or d == 0:
        return {
            "mean": np.zeros((d,), dtype=np.float64),
            "std": np.ones((d,), dtype=np.float64),
            "loadings": np.zeros((d, 0), dtype=np.float64),
            "scores": np.zeros((n, 0), dtype=np.float64),
            "communalities": np.zeros((d,), dtype=np.float64),
            "ss_loadings": np.zeros((0,), dtype=np.float64),
            "factor_proportion": np.zeros((0,), dtype=np.float64),
            "converged": False,
            "iterations": 0,
        }

    k = min(max(0, num_factors), d)
    mean = embeddings.mean(axis=0)
    if n > 1:
        std = embeddings.std(axis=0, ddof=1)
    else:
        std = np.zeros((d,), dtype=np.float64)
    std = np.where(std > 1e-12, std, 1.0)
    z = (embeddings - mean) / std

    if n > 1:
        corr = np.corrcoef(z, rowvar=False)
    else:
        corr = np.eye(d, dtype=np.float64)

    corr = np.nan_to_num(corr, nan=0.0)
    corr = 0.5 * (corr + corr.T)

    communalities = _initial_communalities_from_smc(corr)
    converged = False
    iterations = 0

    loadings = np.zeros((d, k), dtype=np.float64)
    if k > 0:
        for iteration in range(1, max_iter + 1):
            reduced = corr.copy()
            np.fill_diagonal(reduced, communalities)

            eigenvalues, eigenvectors = np.linalg.eigh(reduced)
            order = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]

            positive = np.clip(eigenvalues[:k], a_min=0.0, a_max=None)
            loadings = eigenvectors[:, :k] * np.sqrt(positive)[None, :]

            new_communalities = np.clip((loadings ** 2).sum(axis=1), 0.0, 1.0)
            delta = float(np.max(np.abs(new_communalities - communalities)))
            communalities = new_communalities
            iterations = iteration
            if delta <= tol:
                converged = True
                break

    scores = z @ loadings if k > 0 else np.zeros((n, 0), dtype=np.float64)
    ss_loadings = (loadings ** 2).sum(axis=0) if k > 0 else np.zeros((0,), dtype=np.float64)
    factor_proportion = ss_loadings / max(d, 1)

    return {
        "mean": mean,
        "std": std,
        "loadings": loadings,
        "scores": scores,
        "communalities": communalities,
        "ss_loadings": ss_loadings,
        "factor_proportion": factor_proportion,
        "converged": converged,
        "iterations": iterations,
    }


def _prompt_aggregate(
    rows: list[dict[str, Any]],
    values: np.ndarray,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        group_id = str(row.get("input_group_id") or row.get("sample_id") or idx)
        grouped.setdefault(group_id, []).append(idx)

    aggregates: list[dict[str, Any]] = []
    for group_id, indices in grouped.items():
        vals = values[indices]
        prompt_text = str(rows[indices[0]].get("seed_user_message", ""))
        aggregates.append(
            {
                "input_group_id": group_id,
                "seed_user_message": prompt_text,
                "num_samples": int(len(indices)),
                "mean_projection": float(vals.mean()),
                "std_projection": float(vals.std(ddof=1)) if len(indices) > 1 else 0.0,
            }
        )

    return sorted(
        aggregates,
        key=lambda row: (row["mean_projection"], row["num_samples"], row["input_group_id"]),
    )


def _response_extremes(
    rows: list[dict[str, Any]],
    values: np.ndarray,
    top_n: int,
) -> dict[str, list[dict[str, Any]]]:
    order = np.argsort(values)
    n = min(max(0, top_n), len(rows))

    def _build_entry(idx: int) -> dict[str, Any]:
        row = rows[idx]
        text = str(row.get("assistant_text", ""))
        return {
            "sample_id": row.get("sample_id"),
            "input_group_id": row.get("input_group_id"),
            "assistant_message_id": row.get("assistant_message_id"),
            "assistant_turn_index": row.get("assistant_turn_index"),
            "projection": float(values[idx]),
            "assistant_text_excerpt": text[:280],
        }

    bottom = [_build_entry(int(idx)) for idx in order[:n]]
    top = [_build_entry(int(idx)) for idx in order[-n:][::-1]]
    return {"top": top, "bottom": bottom}


def _build_component_extremes(
    rows: list[dict[str, Any]],
    scores: np.ndarray,
    *,
    top_n: int,
    method_label: str,
) -> dict[str, Any]:
    """Build response/prompt extremes for each component column in scores."""
    result: dict[str, Any] = {"method": method_label, "components": []}
    if scores.ndim != 2 or scores.shape[1] == 0:
        return result

    for component_index in range(scores.shape[1]):
        values = scores[:, component_index]
        prompt_aggregates = _prompt_aggregate(rows, values)
        n = min(max(0, top_n), len(prompt_aggregates))
        prompt_bottom = prompt_aggregates[:n]
        prompt_top = prompt_aggregates[-n:][::-1]

        response_extreme = _response_extremes(rows, values, top_n=top_n)
        result["components"].append(
            {
                "component_index": int(component_index),
                "prompt_extremes": {
                    "top": prompt_top,
                    "bottom": prompt_bottom,
                },
                "response_extremes": response_extreme,
            }
        )

    return result


def _projection_rows(
    rows: list[dict[str, Any]],
    pca_scores: np.ndarray,
    paf_scores: np.ndarray,
) -> list[dict[str, Any]]:
    projection_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        projection_rows.append(
            {
                **row,
                "pca_projections": pca_scores[idx].tolist() if pca_scores.size else [],
                "paf_scores": paf_scores[idx].tolist() if paf_scores.size else [],
            }
        )
    return projection_rows


def run_behavior_decomposition(
    config: BehaviorDecompositionConfig,
) -> tuple[Dataset, BehaviorDecompositionResult]:
    """Run PCA/PAF decomposition and extremity mining over response embeddings."""
    logger = setup_logging()

    init_run(config.run_dir, base_config={"behavior_decomposition": config.model_dump(mode="json")})
    register_stage_fingerprint(
        config.run_dir,
        "behavior_decomposition",
        config.model_dump(mode="json"),
    )

    output_paths = _resolve_output_paths(config)
    if config.overwrite_output:
        for path in output_paths.values():
            if path.exists():
                path.unlink()

    if config.resume and all(path.exists() for path in output_paths.values()) and not config.overwrite_output:
        rows, _ = read_jsonl_tolerant(output_paths["projections"])
        summary = json.loads(output_paths["summary"].read_text(encoding="utf-8"))
        result = BehaviorDecompositionResult(
            pca_path=output_paths["pca"],
            paf_path=output_paths["paf"],
            projections_path=output_paths["projections"],
            summary_path=output_paths["summary"],
            num_samples=int(summary.get("num_samples", len(rows))),
            embedding_dim=int(summary.get("embedding_dim", 0)),
            pca_components=int(summary.get("pca", {}).get("num_components", 0)),
            paf_factors=int(summary.get("paf", {}).get("num_factors", 0)),
        )
        logger.info("Resuming behavior_decomposition from existing artifacts in %s", config.run_dir)
        return Dataset.from_list(rows), result

    rows, embeddings = _load_embedding_inputs(config)
    if embeddings.shape[0] == 0:
        raise ValueError("No embeddings available for decomposition.")

    pca = _compute_pca(embeddings, top_k=config.pca_top_k)
    paf = _compute_paf(
        embeddings,
        num_factors=config.paf_num_factors,
        max_iter=max(1, config.paf_max_iter),
        tol=max(config.paf_tol, 1e-12),
    )

    pca_scores = pca["projections"]
    paf_scores = paf["scores"]

    projection_rows = _projection_rows(rows, pca_scores, paf_scores)
    write_jsonl_atomic(output_paths["projections"], projection_rows)

    np.savez(
        output_paths["pca"],
        mean=pca["mean"],
        eigenvalues=pca["eigenvalues"],
        explained_variance_ratio=pca["explained_variance_ratio"],
        components=pca["components"],
        projections=pca["projections"],
    )
    np.savez(
        output_paths["paf"],
        mean=paf["mean"],
        std=paf["std"],
        loadings=paf["loadings"],
        scores=paf["scores"],
        communalities=paf["communalities"],
        ss_loadings=paf["ss_loadings"],
        factor_proportion=paf["factor_proportion"],
        converged=np.array([1 if paf["converged"] else 0], dtype=np.int32),
        iterations=np.array([int(paf["iterations"])], dtype=np.int32),
    )

    pca_extremes = _build_component_extremes(
        rows,
        pca_scores,
        top_n=config.extremes_top_n,
        method_label="pca",
    )
    paf_extremes = _build_component_extremes(
        rows,
        paf_scores,
        top_n=config.extremes_top_n,
        method_label="paf",
    )

    summary = {
        "created_at": _now_iso(),
        "run_dir": str(config.run_dir),
        "num_samples": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "pca": {
            "num_components": int(pca["components"].shape[0]),
            "eigenvalues": pca["eigenvalues"].tolist(),
            "explained_variance_ratio": pca["explained_variance_ratio"].tolist(),
        },
        "paf": {
            "num_factors": int(paf["loadings"].shape[1]),
            "converged": bool(paf["converged"]),
            "iterations": int(paf["iterations"]),
            "ss_loadings": paf["ss_loadings"].tolist(),
            "factor_proportion": paf["factor_proportion"].tolist(),
        },
        "extremes": {
            "pca": pca_extremes,
            "paf": paf_extremes,
        },
        "paths": {key: str(path) for key, path in output_paths.items()},
    }
    output_paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")

    record_stage_event(
        config.run_dir,
        StageEventRecord(
            event_id=_event_id("behavior_decomposition", "complete", _now_iso()),
            stage="behavior_decomposition",
            event_type="complete",
            created_at=_now_iso(),
            payload={
                "num_samples": int(embeddings.shape[0]),
                "embedding_dim": int(embeddings.shape[1]),
                "pca_components": int(pca["components"].shape[0]),
                "paf_factors": int(paf["loadings"].shape[1]),
            },
        ),
    )

    result = BehaviorDecompositionResult(
        pca_path=output_paths["pca"],
        paf_path=output_paths["paf"],
        projections_path=output_paths["projections"],
        summary_path=output_paths["summary"],
        num_samples=int(embeddings.shape[0]),
        embedding_dim=int(embeddings.shape[1]),
        pca_components=int(pca["components"].shape[0]),
        paf_factors=int(paf["loadings"].shape[1]),
    )

    logger.info(
        "Behavior decomposition complete: samples=%d dim=%d pca_k=%d paf_k=%d",
        result.num_samples,
        result.embedding_dim,
        result.pca_components,
        result.paf_factors,
    )
    return Dataset.from_list(projection_rows), result
