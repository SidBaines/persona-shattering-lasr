"""Model materialization for suite eval runs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from scripts.evals.config import ModelSpec
from scripts.evals.lora_merge import merge_adapters
from scripts.evals.model_resolution import resolve_model_reference


@dataclass(frozen=True)
class MaterializedModel:
    model_name: str
    model_spec_name: str
    model_uri: str
    cache_key: str
    materialized_path: Path | None


def _adapter_ref_for_key(path: str) -> str:
    ref = path
    subfolder: str | None = None
    if "::" in path:
        ref, subfolder = path.split("::", 1)
        subfolder = subfolder or None

    resolved_ref = resolve_model_reference(ref, kind="adapter")
    if subfolder:
        return f"{resolved_ref}::{subfolder}"
    return resolved_ref


def _compute_model_key(model: ModelSpec) -> str:
    resolved_base = resolve_model_reference(model.base_model, kind="base model")
    payload = {
        "base_model": resolved_base,
        "dtype": model.dtype,
        "device_map": model.device_map,
        "adapters": [
            {"path": _adapter_ref_for_key(adapter.path), "scale": adapter.scale}
            for adapter in model.adapters
        ],
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _model_uri_for(base: str | Path) -> str:
    return f"hf/{base}"


def _models_cache_root(output_root: Path) -> Path:
    """Resolve stable cache root for merged models.

    Defaults to a suite-level shared cache directory (sibling of run dirs), so
    repeated runs with different timestamps reuse the same merged artifacts.
    """
    override = os.environ.get("EVALS_MODEL_CACHE_DIR")
    if override:
        return Path(override)
    return output_root.parent / "_models_cache"


def materialize_model(model: ModelSpec, output_root: Path) -> MaterializedModel:
    """Materialize a model spec into an Inspect model URI."""
    cache_key = _compute_model_key(model)
    resolved_base_model = resolve_model_reference(model.base_model, kind="base model")

    if not model.adapters:
        return MaterializedModel(
            model_name=model.base_model,
            model_spec_name=model.name,
            model_uri=_model_uri_for(resolved_base_model),
            cache_key=cache_key,
            materialized_path=None,
        )

    models_root = _models_cache_root(output_root)
    models_root.mkdir(parents=True, exist_ok=True)
    target_dir = models_root / cache_key

    config_path = target_dir / "config.json"
    if not config_path.exists():
        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        try:
            merge_adapters(
                base_model=resolved_base_model,
                adapters=model.adapters,
                output_dir=target_dir,
                dtype=model.dtype,
                device_map=model.device_map,
            )
        except Exception:
            # Remove partial artifacts (for example, interrupted shard writes).
            shutil.rmtree(target_dir, ignore_errors=True)
            raise

    return MaterializedModel(
        model_name=model.base_model,
        model_spec_name=model.name,
        model_uri=_model_uri_for(target_dir),
        cache_key=cache_key,
        materialized_path=target_dir,
    )
