"""Response embedding extraction and variance diagnostics."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import random
import shutil
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
from scripts.unsupervised_runs import build_embedding_slug, resolve_embedding_artifact_paths
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


def _encode_texts_openai(
    texts: list[str],
    config: ResponseEmbeddingConfig,
) -> np.ndarray:
    """Encode response texts with the OpenAI embeddings API."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    from openai import OpenAI

    openai_cfg = config.openai
    logger = setup_logging()
    api_key = os.environ.get(openai_cfg.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Missing {openai_cfg.api_key_env}. Set it to use backend='openai'."
        )

    client = OpenAI(api_key=api_key)
    batch_size = max(1, openai_cfg.batch_size)
    total_texts = len(texts)
    total_batches = (total_texts + batch_size - 1) // batch_size
    start_time = time.perf_counter()

    logger.info(
        "Starting OpenAI embedding encode: samples=%d batches=%d batch_size=%d model=%s",
        total_texts,
        total_batches,
        batch_size,
        openai_cfg.model,
    )

    embeddings: list[np.ndarray] = []
    for batch_index, start in enumerate(range(0, total_texts, batch_size), start=1):
        batch_texts = texts[start : start + batch_size]
        request_kwargs: dict[str, Any] = {
            "model": openai_cfg.model,
            "input": batch_texts,
        }
        if openai_cfg.dimensions is not None:
            request_kwargs["dimensions"] = openai_cfg.dimensions
        response = client.embeddings.create(**request_kwargs)
        batch_matrix = np.array([row.embedding for row in response.data], dtype=np.float32)
        if openai_cfg.normalize and batch_matrix.size:
            norms = np.linalg.norm(batch_matrix, axis=1, keepdims=True)
            batch_matrix = batch_matrix / np.clip(norms, 1e-12, None)
        embeddings.append(batch_matrix)

        processed = min(batch_index * batch_size, total_texts)
        elapsed = time.perf_counter() - start_time
        rate = processed / elapsed if elapsed > 0 else 0.0
        remaining = total_texts - processed
        eta_seconds = remaining / rate if rate > 0 else 0.0
        pct = (processed / total_texts) * 100.0 if total_texts else 100.0
        logger.info(
            "OpenAI embedding progress: batch %d/%d, samples %d/%d (%.1f%%), elapsed %.1fs, ETA %.1fs",
            batch_index,
            total_batches,
            processed,
            total_texts,
            pct,
            elapsed,
            eta_seconds,
        )

    matrix = np.vstack(embeddings)
    logger.info(
        "Encoded %d texts with OpenAI model=%s (dim=%d)",
        matrix.shape[0],
        openai_cfg.model,
        matrix.shape[1],
    )
    return matrix


def _is_retryable_openai_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and status_code in {408, 409, 429, 500, 502, 503, 504}:
        return True

    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    if isinstance(response_status, int) and response_status in {408, 409, 429, 500, 502, 503, 504}:
        return True

    name = exc.__class__.__name__.lower()
    return any(
        token in name
        for token in (
            "ratelimit",
            "timeout",
            "apiconnection",
            "internalserver",
        )
    )


def _call_with_retry(
    fn,
    *,
    should_retry,
    max_retries: int,
    initial_backoff_seconds: float,
    max_backoff_seconds: float,
    logger,
    context: str,
):
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001
            if attempt >= max_retries or not should_retry(exc):
                raise
            backoff = min(
                max_backoff_seconds,
                initial_backoff_seconds * (2 ** (attempt - 1)),
            )
            jitter = random.uniform(0.0, min(1.0, backoff * 0.1))
            sleep_seconds = backoff + jitter
            logger.warning(
                "%s failed with %s on attempt %d/%d; retrying in %.1fs",
                context,
                exc.__class__.__name__,
                attempt,
                max_retries,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)


class _EmbeddingBatchEncoder:
    """Backend-specific batch encoder interface."""

    batch_size: int

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        """Release any backend resources."""


class _LocalHFEmbeddingBatchEncoder(_EmbeddingBatchEncoder):
    """Incremental local Hugging Face embedding encoder."""

    def __init__(self, config: ResponseEmbeddingConfig) -> None:
        self._config = config
        self._logger = setup_logging()
        local_cfg = config.local_hf

        dtype = getattr(torch, local_cfg.dtype, None)
        if dtype is None:
            raise ValueError(f"Unsupported local_hf dtype: {local_cfg.dtype}")
        if not torch.cuda.is_available() and dtype in {torch.bfloat16, torch.float16}:
            self._logger.warning(
                "CUDA not available; falling back embedding dtype from %s to float32.",
                local_cfg.dtype,
            )
            dtype = torch.float32

        self._tokenizer = AutoTokenizer.from_pretrained(
            local_cfg.model,
            revision=local_cfg.revision,
            use_fast=True,
            trust_remote_code=local_cfg.trust_remote_code,
        )
        self._model = AutoModel.from_pretrained(
            local_cfg.model,
            revision=local_cfg.revision,
            torch_dtype=dtype,
            device_map=local_cfg.device_map,
            trust_remote_code=local_cfg.trust_remote_code,
        )
        self._model.eval()
        self._device = next(self._model.parameters()).device
        self.batch_size = max(1, local_cfg.batch_size)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        local_cfg = self._config.local_hf
        tokens = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=local_cfg.max_length,
            return_tensors="pt",
        )
        tokens = {key: value.to(self._device) for key, value in tokens.items()}
        with torch.no_grad():
            output = self._model(**tokens)
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
            return pooled.detach().cpu().to(torch.float32).numpy()

    def close(self) -> None:
        del self._model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class _OpenAIEmbeddingBatchEncoder(_EmbeddingBatchEncoder):
    """Incremental OpenAI embedding encoder with retry/backoff."""

    def __init__(self, config: ResponseEmbeddingConfig) -> None:
        from openai import OpenAI

        self._config = config
        self._logger = setup_logging()
        openai_cfg = config.openai
        api_key = os.environ.get(openai_cfg.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing {openai_cfg.api_key_env}. Set it to use backend='openai'."
            )

        self._client = OpenAI(api_key=api_key)
        self.batch_size = max(1, openai_cfg.batch_size)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        openai_cfg = self._config.openai

        def _request():
            request_kwargs: dict[str, Any] = {
                "model": openai_cfg.model,
                "input": texts,
            }
            if openai_cfg.dimensions is not None:
                request_kwargs["dimensions"] = openai_cfg.dimensions
            return self._client.embeddings.create(**request_kwargs)

        response = _call_with_retry(
            _request,
            should_retry=_is_retryable_openai_error,
            max_retries=max(1, openai_cfg.max_retries),
            initial_backoff_seconds=max(0.0, openai_cfg.initial_backoff_seconds),
            max_backoff_seconds=max(0.0, openai_cfg.max_backoff_seconds),
            logger=self._logger,
            context=f"OpenAI embeddings batch model={openai_cfg.model}",
        )
        batch_matrix = np.array([row.embedding for row in response.data], dtype=np.float32)
        if openai_cfg.normalize and batch_matrix.size:
            norms = np.linalg.norm(batch_matrix, axis=1, keepdims=True)
            batch_matrix = batch_matrix / np.clip(norms, 1e-12, None)
        return batch_matrix


def _create_batch_encoder(config: ResponseEmbeddingConfig) -> _EmbeddingBatchEncoder:
    if config.backend == "local_hf":
        return _LocalHFEmbeddingBatchEncoder(config)
    if config.backend == "openai":
        return _OpenAIEmbeddingBatchEncoder(config)
    raise ValueError(f"Unsupported embeddings backend: {config.backend}")


def _checkpoint_dir(output_paths: dict[str, Path]) -> Path:
    return output_paths["artifact_dir"] / "_embedding_checkpoint"


def _checkpoint_state_path(output_paths: dict[str, Path]) -> Path:
    return _checkpoint_dir(output_paths) / "state.json"


def _checkpoint_batch_path(output_paths: dict[str, Path], batch_index: int) -> Path:
    return _checkpoint_dir(output_paths) / f"batch_{batch_index:06d}.npy"


def _write_checkpoint_state(
    output_paths: dict[str, Path],
    *,
    artifact_slug: str,
    backend: str,
    batch_size: int,
    total_rows: int,
    completed_rows: int,
    completed_batches: int,
    embedding_dim: int | None,
) -> None:
    checkpoint_dir = _checkpoint_dir(output_paths)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_slug": artifact_slug,
        "backend": backend,
        "batch_size": int(batch_size),
        "total_rows": int(total_rows),
        "completed_rows": int(completed_rows),
        "completed_batches": int(completed_batches),
        "embedding_dim": int(embedding_dim) if embedding_dim is not None else None,
    }
    _checkpoint_state_path(output_paths).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_checkpoint_state(output_paths: dict[str, Path]) -> dict[str, Any] | None:
    path = _checkpoint_state_path(output_paths)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_checkpoint_batches(output_paths: dict[str, Path], completed_batches: int) -> list[np.ndarray]:
    batches: list[np.ndarray] = []
    for batch_index in range(completed_batches):
        batch_path = _checkpoint_batch_path(output_paths, batch_index)
        if not batch_path.exists():
            raise FileNotFoundError(
                f"Checkpoint state expects batch {batch_index}, but file is missing: {batch_path}"
            )
        batches.append(np.load(batch_path))
    return batches


def _resolve_output_paths(config: ResponseEmbeddingConfig) -> dict[str, Path]:
    return resolve_embedding_artifact_paths(
        config.run_dir,
        _resolved_artifact_slug(config),
        output_prefix=config.output_prefix,
    )


def _resolved_artifact_slug(config: ResponseEmbeddingConfig) -> str:
    if config.artifact_slug:
        return config.artifact_slug
    if config.backend == "openai":
        model = config.openai.model
        normalize = config.openai.normalize
        max_length = 0
    else:
        model = config.local_hf.model
        normalize = config.local_hf.normalize
        max_length = config.local_hf.max_length
    return build_embedding_slug(
        model=model,
        analysis_unit=config.analysis_unit,
        normalize=normalize,
        max_length=max_length,
        target_variant=config.target_variant,
    )


def _stage_name(config: ResponseEmbeddingConfig) -> str:
    return f"response_embeddings:{_resolved_artifact_slug(config)}"


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

    config_payload = config.model_dump(mode="json")
    config_for_fingerprint = {
        key: value
        for key, value in config_payload.items()
        if key not in {"resume", "overwrite_output"}
    }
    init_run(config.run_dir, base_config={"response_embeddings": config_payload})
    register_stage_fingerprint(
        config.run_dir,
        _stage_name(config),
        config_for_fingerprint,
    )

    output_paths = _resolve_output_paths(config)
    output_paths["artifact_dir"].mkdir(parents=True, exist_ok=True)
    persisted_outputs = {
        key: path for key, path in output_paths.items() if key != "artifact_dir"
    }
    all_outputs_exist = all(path.exists() for path in persisted_outputs.values())

    if config.overwrite_output:
        for path in persisted_outputs.values():
            if path.exists():
                path.unlink()
        checkpoint_dir = _checkpoint_dir(output_paths)
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)

    if config.resume and all_outputs_exist and not config.overwrite_output:
        rows, _ = read_jsonl_tolerant(output_paths["metadata"])
        loaded = np.load(output_paths["embeddings"])
        logger.info("Resuming response_embeddings from existing artifacts in %s", config.run_dir)
        result = ResponseEmbeddingResult(
            artifact_slug=_resolved_artifact_slug(config),
            artifact_dir=output_paths["artifact_dir"],
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
    for idx, row in enumerate(rows):
        row["embedding_index"] = idx

    write_jsonl_atomic(output_paths["metadata"], rows)

    encoder = _create_batch_encoder(config)
    batch_size = encoder.batch_size
    total_texts = len(texts)
    checkpoint_state = _load_checkpoint_state(output_paths) if config.resume else None
    completed_rows = 0
    completed_batches = 0
    batches: list[np.ndarray] = []

    if checkpoint_state is not None:
        state_total_rows = int(checkpoint_state.get("total_rows", -1))
        if state_total_rows != total_texts:
            raise ValueError(
                "Checkpoint row count does not match current embedding input. "
                f"checkpoint={state_total_rows} current={total_texts}"
            )
        completed_rows = int(checkpoint_state.get("completed_rows", 0))
        completed_batches = int(checkpoint_state.get("completed_batches", 0))
        batches = _load_checkpoint_batches(output_paths, completed_batches)
        logger.info(
            "Resuming embedding checkpoint for %s: completed_rows=%d/%d completed_batches=%d",
            _resolved_artifact_slug(config),
            completed_rows,
            total_texts,
            completed_batches,
        )
    else:
        _write_checkpoint_state(
            output_paths,
            artifact_slug=_resolved_artifact_slug(config),
            backend=config.backend,
            batch_size=batch_size,
            total_rows=total_texts,
            completed_rows=0,
            completed_batches=0,
            embedding_dim=None,
        )

    start_time = time.perf_counter()
    embedding_dim: int | None = None
    try:
        for batch_index, start in enumerate(
            range(completed_rows, total_texts, batch_size),
            start=completed_batches,
        ):
            batch_texts = texts[start : start + batch_size]
            batch_matrix = encoder.encode_batch(batch_texts)
            if batch_matrix.ndim != 2:
                raise ValueError(
                    f"Expected 2D embedding batch, got shape={batch_matrix.shape}"
                )
            if batch_matrix.shape[0] != len(batch_texts):
                raise ValueError(
                    "Embedding batch row count mismatch. "
                    f"batch_texts={len(batch_texts)} embeddings={batch_matrix.shape[0]}"
                )
            if embedding_dim is None:
                embedding_dim = int(batch_matrix.shape[1])
            elif int(batch_matrix.shape[1]) != embedding_dim:
                raise ValueError(
                    "Embedding dimension changed across batches. "
                    f"expected={embedding_dim} actual={batch_matrix.shape[1]}"
                )

            np.save(_checkpoint_batch_path(output_paths, batch_index), batch_matrix)
            batches.append(batch_matrix)

            completed_rows = min(start + len(batch_texts), total_texts)
            completed_batches = batch_index + 1
            _write_checkpoint_state(
                output_paths,
                artifact_slug=_resolved_artifact_slug(config),
                backend=config.backend,
                batch_size=batch_size,
                total_rows=total_texts,
                completed_rows=completed_rows,
                completed_batches=completed_batches,
                embedding_dim=embedding_dim,
            )

            elapsed = time.perf_counter() - start_time
            rate = completed_rows / elapsed if elapsed > 0 else 0.0
            remaining = total_texts - completed_rows
            eta_seconds = remaining / rate if rate > 0 else 0.0
            pct = (completed_rows / total_texts) * 100.0 if total_texts else 100.0
            logger.info(
                "Embedding checkpoint progress: batch %d, samples %d/%d (%.1f%%), elapsed %.1fs, ETA %.1fs",
                completed_batches,
                completed_rows,
                total_texts,
                pct,
                elapsed,
                eta_seconds,
            )
    finally:
        encoder.close()

    embeddings = np.vstack(batches) if batches else np.zeros((0, 0), dtype=np.float32)
    if embeddings.shape[0] != len(rows):
        raise RuntimeError(
            "Embedding row count mismatch. "
            f"embeddings={embeddings.shape[0]} metadata_rows={len(rows)}"
        )

    np.save(output_paths["embeddings"], embeddings)
    variance_report = _compute_variance_report(rows, embeddings)
    output_paths["variance"].write_text(json.dumps(variance_report, indent=2), encoding="utf-8")

    manifest = {
        "created_at": _now_iso(),
        "run_dir": str(config.run_dir),
        "artifact_slug": _resolved_artifact_slug(config),
        "artifact_dir": str(output_paths["artifact_dir"]),
        "analysis_unit": config.analysis_unit,
        "target_variant": config.target_variant,
        "backend": config.backend,
        "local_hf": config.local_hf.model_dump(mode="json"),
        "openai": config.openai.model_dump(mode="json"),
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
                "artifact_slug": _resolved_artifact_slug(config),
                "num_samples": int(embeddings.shape[0]),
                "embedding_dim": int(embeddings.shape[1]),
                "analysis_unit": config.analysis_unit,
                "backend": config.backend,
            },
        ),
    )

    result = ResponseEmbeddingResult(
        artifact_slug=_resolved_artifact_slug(config),
        artifact_dir=output_paths["artifact_dir"],
        metadata_path=output_paths["metadata"],
        embeddings_path=output_paths["embeddings"],
        variance_path=output_paths["variance"],
        manifest_path=output_paths["manifest"],
        num_samples=int(embeddings.shape[0]),
        embedding_dim=int(embeddings.shape[1]),
        analysis_unit=config.analysis_unit,
    )
    checkpoint_dir = _checkpoint_dir(output_paths)
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    return Dataset.from_list(rows), result
