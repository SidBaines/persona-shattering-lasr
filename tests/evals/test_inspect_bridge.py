"""Tests for inspect task resolution helpers."""

from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset, Sample

from scripts.evals.inspect_bridge import resolve_inspect_task_ref
from scripts.evals.inspect_bridge import _with_unique_sample_ids


def test_resolve_inspect_task_ref_mmlu_alias():
    assert resolve_inspect_task_ref("mmlu") == "inspect_evals/mmlu_0_shot"


def test_resolve_inspect_task_ref_mmlu_legacy_path():
    assert (
        resolve_inspect_task_ref("inspect_evals/mmlu")
        == "inspect_evals/mmlu_0_shot"
    )


def test_resolve_inspect_task_ref_mmlu_pro_alias():
    assert resolve_inspect_task_ref("mmlu_pro") == "inspect_evals/mmlu_pro"


def test_with_unique_sample_ids_rewrites_duplicates():
    task = Task(
        dataset=MemoryDataset(
            samples=[
                Sample(input="q1", target="a", id="dup"),
                Sample(input="q2", target="b", id="dup"),
                Sample(input="q3", target="c", id="dup"),
            ]
        )
    )

    fixed_task, rewritten = _with_unique_sample_ids(task)
    ids = [sample.id for sample in fixed_task.dataset]

    assert rewritten == 2
    assert ids == ["dup", "dup__dup1", "dup__dup2"]


def test_with_unique_sample_ids_noop_when_unique():
    task = Task(
        dataset=MemoryDataset(
            samples=[
                Sample(input="q1", target="a", id="a"),
                Sample(input="q2", target="b", id="b"),
            ]
        )
    )

    fixed_task, rewritten = _with_unique_sample_ids(task)
    ids = [sample.id for sample in fixed_task.dataset]

    assert rewritten == 0
    assert ids == ["a", "b"]
