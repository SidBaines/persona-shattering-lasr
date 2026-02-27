"""Tests for benchmark dispatch aliases."""

from __future__ import annotations

from types import SimpleNamespace

from inspect_ai.dataset import MemoryDataset, Sample

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


def test_trait_filter_restricts_dataset(monkeypatch) -> None:
    task = SimpleNamespace(
        dataset=MemoryDataset(
            samples=[
                Sample(
                    input="q1",
                    target=["A"],
                    choices=["a"],
                    metadata={"trait": "Openness"},
                ),
                Sample(
                    input="q2",
                    target=["A"],
                    choices=["a"],
                    metadata={"trait": "Extraversion"},
                ),
                Sample(
                    input="q3",
                    target=["A"],
                    choices=["a"],
                    metadata={"trait": "Extraversion"},
                ),
            ],
            name="trait",
        ),
    )

    monkeypatch.setattr(
        "inspect_evals.personality.personality.personality_TRAIT",
        lambda **kwargs: task,
    )

    spec = InspectBenchmarkSpec(
        name="trait_eval",
        benchmark="personality_trait",
        benchmark_args={
            "personality": "high extraversion",
            "trait": "Extraversion",
            "shuffle": "questions",
        },
        limit=2,
    )
    filtered_task = build_benchmark_task(spec)

    assert len(filtered_task.dataset) == 2
    assert all(
        sample.metadata["trait"] == "Extraversion"
        for sample in filtered_task.dataset
    )


def test_trait_builder_uses_single_split_fast_path(monkeypatch) -> None:
    captured = {}

    monkeypatch.setattr(
        "scripts.evals.inspect_benchmarks._build_single_trait_task",
        lambda *, personality, trait: captured.update(
            {"personality": personality, "trait": trait}
        ) or "single-trait-task",
    )
    monkeypatch.setattr(
        "inspect_evals.personality.personality.personality_TRAIT",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not be called")),
    )

    spec = InspectBenchmarkSpec(
        name="trait_eval",
        benchmark="personality_trait",
        benchmark_args={
            "personality": "high neuroticism",
            "trait": "Neuroticism",
        },
    )

    task = build_benchmark_task(spec)
    assert task == "single-trait-task"
    assert captured == {
        "personality": "high neuroticism",
        "trait": "Neuroticism",
    }
