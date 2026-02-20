"""Tests for model/adaptor reference resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.evals.model_resolution import resolve_model_reference


def test_resolve_local_path(tmp_path: Path):
    local_dir = tmp_path / "model_dir"
    local_dir.mkdir()
    resolved = resolve_model_reference(str(local_dir), kind="base model")
    assert resolved == str(local_dir.resolve())


def test_ambiguous_local_and_hf_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    local_ref = Path("org/model")
    local_ref.mkdir(parents=True)

    monkeypatch.setattr(
        "scripts.evals.model_resolution._hf_repo_exists",
        lambda _: True,
    )

    with pytest.raises(ValueError, match="Ambiguous base model reference"):
        resolve_model_reference("org/model", kind="base model")


def test_explicit_local_missing_raises():
    with pytest.raises(FileNotFoundError):
        resolve_model_reference("local://does/not/exist", kind="adapter")
