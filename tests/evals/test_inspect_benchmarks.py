"""Tests for benchmark dispatch aliases."""

from __future__ import annotations

from scripts.evals.config import InspectBenchmarkSpec
from scripts.evals.inspect_benchmarks import _canonical_name, build_benchmark_task


def test_canonical_name_keeps_existing_aliases() -> None:
    assert _canonical_name("truthfulqagen") == "truthfulqa"
    assert _canonical_name("gpqa_diamond") == "gpqa"
    assert _canonical_name("pop_qa") == "popqa"


def test_trait_alias_dispatch(monkeypatch) -> None:
    captured = {}

    def _fake_trait(**kwargs):
        captured["kwargs"] = kwargs
        return "trait-task"

    monkeypatch.setattr(
        "inspect_evals.personality.personality.personality_TRAIT",
        _fake_trait,
    )

    spec = InspectBenchmarkSpec(
        name="trait_eval",
        benchmark="trait",
        benchmark_args={"personality": "high openness", "seed": 7},
    )
    task = build_benchmark_task(spec)
    assert task == "trait-task"
    assert captured["kwargs"] == {"personality": "high openness", "seed": 7}


def test_bfi_alias_dispatch(monkeypatch) -> None:
    captured = {}

    def _fake_bfi(**kwargs):
        captured["kwargs"] = kwargs
        return "bfi-task"

    monkeypatch.setattr(
        "inspect_evals.personality.personality.personality_BFI",
        _fake_bfi,
    )

    spec = InspectBenchmarkSpec(
        name="bfi_eval",
        benchmark="personality_bfi",
        benchmark_args={"personality": "high agreeableness"},
    )
    task = build_benchmark_task(spec)
    assert task == "bfi-task"
    assert captured["kwargs"] == {"personality": "high agreeableness"}

