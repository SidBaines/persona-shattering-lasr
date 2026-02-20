"""Tests for shared LoRA composition helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.evals.config import AdapterConfig
from scripts.utils.lora_composition import (
    delete_materialized_model_dir,
    normalize_weighted_adapters,
    parse_weighted_adapter,
    split_adapter_reference,
)


def test_parse_weighted_adapter_defaults_scale() -> None:
    parsed = parse_weighted_adapter("local://scratch/a")
    assert parsed.path == "local://scratch/a"
    assert parsed.scale == 1.0


def test_parse_weighted_adapter_with_scale() -> None:
    parsed = parse_weighted_adapter("hf://org/adapter@-0.25")
    assert parsed.path == "hf://org/adapter"
    assert parsed.scale == -0.25


def test_parse_weighted_adapter_invalid_scale() -> None:
    with pytest.raises(ValueError, match="Invalid adapter scale"):
        parse_weighted_adapter("hf://org/adapter@abc")


def test_split_adapter_reference_with_subfolder() -> None:
    ref, subfolder = split_adapter_reference("hf://org/adapter::adapter")
    assert ref == "hf://org/adapter"
    assert subfolder == "adapter"


def test_normalize_weighted_adapters_accepts_adapter_config() -> None:
    adapters = normalize_weighted_adapters(
        [AdapterConfig(path="local://scratch/a", scale=0.5)]
    )
    assert len(adapters) == 1
    assert adapters[0].path == "local://scratch/a"
    assert adapters[0].scale == 0.5


def test_delete_materialized_model_dir_prunes_parent(tmp_path: Path) -> None:
    parent = tmp_path / "_models_cache"
    target = parent / "abc123"
    target.mkdir(parents=True)
    (target / "config.json").write_text("{}", encoding="utf-8")

    delete_materialized_model_dir(target, prune_empty_parent=True)

    assert not target.exists()
    assert not parent.exists()
