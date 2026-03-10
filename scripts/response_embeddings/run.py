"""Response embedding extraction and variance diagnostics."""

from __future__ import annotations

import gc
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModel, AutoTokenizer

from scripts.datasets import (
    get_run_paths,
    init_run,
    load_samples,
    materialize_canonical_samples,
    record_stage_event,
    register_stage_fingerprint,
    render_messages,
)
from scripts.datasets.io import read_jsonl_tolerant, write_jsonl_atomic
from scripts.datasets.schema import StageEventRecord
from scripts.response_embeddings.config import ResponseEmbeddingConfig, ResponseEmbeddingResult
from scripts.utils import setup_logging


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _event_id(*parts: str) -> str:
    text = ":".join(parts)
    return f"evt_{hashlib.sha256(text.encode('utf-8')).hexdigest()[:24]}"


def _extract_assistant_candidates(config: ResponseEmbeddingConfig) -> list[dict[str, Any]]:
    """Extract assistant-response candidates from canonical run rows."""
    materialize_canonical_samples(config.run_dir)
    samples = load_samples(config.run_dir)

    candidates: list[dict[str, Any]] = []
    for sample in samples:
        effective_messages = render_messages(sample, config.target_variant)
        user_positions = [idx for idx, msg in enumerate(effective_messages) if msg.role == "user"]
        seed_user_message = effective_messages[user_positions[0]].content if user_positions else ""

        assistant_positions = [
            idx for idx, msg in enumerate(effective_messages) if msg.role == "assistant"
        ]
        if not assistant_positions:
            continue

        if config.analysis_unit == "assistant_final_turn":
            selected_positions = [assistant_positions[-1]]
        elif config.analysis_unit == "assistant_first_turn":
            selected_positions = [assistant_positions[0]]
        else:
            selected_positions = assistant_positions

        for assistant_turn_index, msg_pos in enumerate(assistant_positions):
            if msg_pos not in selected_positions:
                continue
            assistant_message = effective_messages[msg_pos]
            preceding_user_message = next(
                (
                    effective_messages[i].content
                    for i in range(msg_pos - 1, -1, -1)
                    if effective_messages[i].role == "user"
                ),
                "",
            )
            metadata_turn = (assistant_message.message_metadata or {}).get("turn_index")
            if isinstance(metadata_turn, int) and metadata_turn >= 0:
                turn_index = metadata_turn
            else:
                turn_index = assistant_turn_index

            candidates.append(
                {
                    "sample_id": sample.sample_id,
                    "input_group_id": sample.input_group_id or sample.sample_id,
                    "response_index": sample.response_index,
                    "assistant_turn_index": int(turn_index),
                    "assistant_message_id": assistant_message.message_id,
                    "candidate_ref": f"assistant_message:{assistant_message.message_id}",
                    "seed_user_message": seed_user_message,
                    "preceding_user_message": preceding_user_message,
                    "assistant_text": assistant_message.content,
                    "analysis_unit": config.analysis_unit,
                }
            )

    return candidates


def _mean_pool_hidden(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool token embeddings with attention-mask weighting."""
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _encode_texts_local_hf(
    texts: list[str],
    config: ResponseEmbeddingConfig,
) -> np.ndarray:
    """Encode response texts with a local HuggingFace embedding model."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    local_cfg = config.local_hf
    logger = setup_logging()

    dtype = getattr(torch, local_cfg.dtype, None)
    if dtype is None:
        raise ValueError(f"Unsupported local_hf dtype: {local_cfg.dtype}")
    if not torch.cuda.is_available() and dtype in {torch.bfloat16, torch.float16}:
        logger.warning(
            "CUDA not available; falling back embedding dtype from %s to float32.",
            local_cfg.dtype,
        )
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        local_cfg.model,
        revision=local_cfg.revision,
        use_fast=True,
        trust_remote_code=local_cfg.trust_remote_code,
    )
    model = AutoModel.from_pretrained(
        local_cfg.model,
        revision=local_cfg.revision,
        torch_dtype=dtype,
        device_map=local_cfg.device_map,
        trust_remote_code=local_cfg.trust_remote_code,
    )

    model.eval()
    device = next(model.parameters()).device

    batch_size = max(1, local_cfg.batch_size)
    total_texts = len(texts)
    total_batches = (total_texts + batch_size - 1) // batch_size
    start_time = time.perf_counter()

    logger.info(
        "Starting embedding encode: samples=%d batches=%d batch_size=%d model=%s",
        total_texts,
        total_batches,
        batch_size,
        local_cfg.model,
    )

    embeddings: list[np.ndarray] = []
    for batch_index, start in enumerate(range(0, total_texts, batch_size), start=1):
        batch_texts = texts[start : start + batch_size]
        tokens = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=local_cfg.max_length,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with torch.no_grad():
            output = model(**tokens)
            if hasattr(output, "last_hidden_state"):
                pooled = _mean_pool_hidden(output.last_hidden_state, tokens["attention_mask"])
            elif isinstance(output, tuple) and output:
                pooled = _mean_pool_hidden(output[0], tokens["attention_mask"])
            else:
                raise ValueError(
                    "Embedding model output does not contain last_hidden_state; "
                    "cannot derive sentence embeddings."
                )

            if local_cfg.normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings.append(pooled.detach().cpu().to(torch.float32).numpy())

        processed = min(batch_index * batch_size, total_texts)
        elapsed = time.perf_counter() - start_time
        rate = processed / elapsed if elapsed > 0 else 0.0
        remaining = total_texts - processed
        eta_seconds = remaining / rate if rate > 0 else 0.0
        pct = (processed / total_texts) * 100.0 if total_texts else 100.0
        logger.info(
            "Embedding progress: batch %d/%d, samples %d/%d (%.1f%%), elapsed %.1fs, ETA %.1fs",
            batch_index,
            total_batches,
            processed,
            total_texts,
            pct,
            elapsed,
            eta_seconds,
        )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    matrix = np.vstack(embeddings)
    logger.info(
        "Encoded %d texts with model=%s (dim=%d)",
        matrix.shape[0],
        local_cfg.model,
        matrix.shape[1],
    )
    return matrix


def _compute_variance_report(
    rows: list[dict[str, Any]],
    embeddings: np.ndarray,
) -> dict[str, Any]:
    """Compute global and per-prompt variance diagnostics."""
    n = int(embeddings.shape[0])
    d = int(embeddings.shape[1]) if embeddings.ndim == 2 else 0
    if n == 0 or d == 0:
        return {
            "global": {
                "num_samples": n,
                "embedding_dim": d,
                "total_variance": 0.0,
                "mean_dimension_variance": 0.0,
                "max_dimension_variance": 0.0,
                "min_dimension_variance": 0.0,
                "mean_embedding_norm": 0.0,
                "centroid_norm": 0.0,
            },
            "per_prompt": [],
            "top_prompts_by_total_variance": [],
        }

    if n > 1:
        per_dim_var = embeddings.var(axis=0, ddof=1)
    else:
        per_dim_var = np.zeros(d, dtype=np.float64)

    global_stats = {
        "num_samples": n,
        "embedding_dim": d,
        "total_variance": float(per_dim_var.sum()),
        "mean_dimension_variance": float(per_dim_var.mean()),
        "max_dimension_variance": float(per_dim_var.max(initial=0.0)),
        "min_dimension_variance": float(per_dim_var.min(initial=0.0)),
        "mean_embedding_norm": float(np.linalg.norm(embeddings, axis=1).mean()),
        "centroid_norm": float(np.linalg.norm(embeddings.mean(axis=0))),
    }

    grouped_indices: dict[str, list[int]] = {}
    prompt_text_by_group: dict[str, str] = {}
    for idx, row in enumerate(rows):
        group_id = str(row.get("input_group_id") or row.get("sample_id") or idx)
        grouped_indices.setdefault(group_id, []).append(idx)
        prompt_text_by_group.setdefault(group_id, str(row.get("seed_user_message", "")))

    per_prompt: list[dict[str, Any]] = []
    for group_id, indices in grouped_indices.items():
        group_embeddings = embeddings[indices]
        if len(indices) > 1:
            group_var = group_embeddings.var(axis=0, ddof=1)
        else:
            group_var = np.zeros(d, dtype=np.float64)
        per_prompt.append(
            {
                "input_group_id": group_id,
                "seed_user_message": prompt_text_by_group.get(group_id, ""),
                "num_samples": int(len(indices)),
                "total_variance": float(group_var.sum()),
                "mean_dimension_variance": float(group_var.mean()),
            }
        )

    per_prompt_sorted = sorted(
        per_prompt,
        key=lambda row: (row["total_variance"], row["num_samples"], row["input_group_id"]),
        reverse=True,
    )

    return {
        "global": global_stats,
        "per_prompt": per_prompt_sorted,
        "top_prompts_by_total_variance": per_prompt_sorted[:20],
    }


def _resolve_output_paths(config: ResponseEmbeddingConfig) -> dict[str, Path]:
    reports_dir = get_run_paths(config.run_dir)["reports_dir"]
    prefix = config.output_prefix
    return {
        "metadata": reports_dir / f"{prefix}_metadata.jsonl",
        "embeddings": reports_dir / f"{prefix}_embeddings.npy",
        "variance": reports_dir / f"{prefix}_variance.json",
        "manifest": reports_dir / f"{prefix}_manifest.json",
    }


def run_response_embeddings(
    config: ResponseEmbeddingConfig,
    dataset: Dataset | None = None,
) -> tuple[Dataset, ResponseEmbeddingResult]:
    """Extract assistant responses, embed them, and compute variance diagnostics."""
    logger = setup_logging()

    if dataset is not None:
        raise ValueError(
            "run_response_embeddings currently supports canonical run-dir input only. "
            "Pass dataset=None and set config.run_dir."
        )

    init_run(config.run_dir, base_config={"response_embeddings": config.model_dump(mode="json")})
    register_stage_fingerprint(
        config.run_dir,
        "response_embeddings",
        config.model_dump(mode="json"),
    )

    output_paths = _resolve_output_paths(config)
    all_outputs_exist = all(path.exists() for path in output_paths.values())

    if config.overwrite_output:
        for path in output_paths.values():
            if path.exists():
                path.unlink()

    if config.resume and all_outputs_exist and not config.overwrite_output:
        rows, _ = read_jsonl_tolerant(output_paths["metadata"])
        loaded = np.load(output_paths["embeddings"])
        logger.info("Resuming response_embeddings from existing artifacts in %s", config.run_dir)
        result = ResponseEmbeddingResult(
            metadata_path=output_paths["metadata"],
            embeddings_path=output_paths["embeddings"],
            variance_path=output_paths["variance"],
            manifest_path=output_paths["manifest"],
            num_samples=int(loaded.shape[0]) if loaded.ndim >= 1 else 0,
            embedding_dim=int(loaded.shape[1]) if loaded.ndim == 2 else 0,
            analysis_unit=config.analysis_unit,
        )
        return Dataset.from_list(rows), result

    rows = _extract_assistant_candidates(config)
    if not rows:
        raise ValueError(
            "No assistant responses found for embedding extraction. "
            "Check rollout outputs and analysis_unit selection."
        )

    texts = [str(row["assistant_text"]) for row in rows]
    if config.backend != "local_hf":
        raise ValueError(f"Unsupported embeddings backend: {config.backend}")
    embeddings = _encode_texts_local_hf(texts, config)

    if embeddings.shape[0] != len(rows):
        raise RuntimeError(
            "Embedding row count mismatch. "
            f"embeddings={embeddings.shape[0]} metadata_rows={len(rows)}"
        )

    for idx, row in enumerate(rows):
        row["embedding_index"] = idx

    variance_report = _compute_variance_report(rows, embeddings)

    write_jsonl_atomic(output_paths["metadata"], rows)
    np.save(output_paths["embeddings"], embeddings)
    output_paths["variance"].write_text(json.dumps(variance_report, indent=2), encoding="utf-8")

    manifest = {
        "created_at": _now_iso(),
        "run_dir": str(config.run_dir),
        "analysis_unit": config.analysis_unit,
        "target_variant": config.target_variant,
        "backend": config.backend,
        "local_hf": config.local_hf.model_dump(mode="json"),
        "num_samples": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "paths": {key: str(path) for key, path in output_paths.items()},
    }
    output_paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    record_stage_event(
        config.run_dir,
        StageEventRecord(
            event_id=_event_id("response_embeddings", "complete", _now_iso()),
            stage="response_embeddings",
            event_type="complete",
            created_at=_now_iso(),
            payload={
                "num_samples": int(embeddings.shape[0]),
                "embedding_dim": int(embeddings.shape[1]),
                "analysis_unit": config.analysis_unit,
                "backend": config.backend,
            },
        ),
    )

    result = ResponseEmbeddingResult(
        metadata_path=output_paths["metadata"],
        embeddings_path=output_paths["embeddings"],
        variance_path=output_paths["variance"],
        manifest_path=output_paths["manifest"],
        num_samples=int(embeddings.shape[0]),
        embedding_dim=int(embeddings.shape[1]),
        analysis_unit=config.analysis_unit,
    )
    return Dataset.from_list(rows), result
