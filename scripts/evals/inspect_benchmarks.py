"""Benchmark task builders for Inspect-based eval runs."""

from __future__ import annotations

import json
import os
from typing import Any

from datasets import load_dataset
from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import includes
from inspect_ai.solver import generate

from scripts.evals.config import InspectBenchmarkSpec


def _build_popqa_task(limit: int | None = None) -> Task:
    ds = load_dataset("akariasai/PopQA", split="test")
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    samples: list[Sample] = []
    for row in ds:
        answers = json.loads(row["possible_answers"])
        samples.append(
            Sample(
                input=row["question"],
                target=answers,
                metadata={
                    "question": row["question"],
                    "id": row.get("id"),
                },
            )
        )

    return Task(
        name="popqa",
        dataset=MemoryDataset(samples=samples, name="popqa"),
        solver=[generate()],
        scorer=includes(),
    )


def _canonical_name(name: str) -> str:
    normalized = "".join(ch for ch in name.lower() if ch.isalnum())
    aliases = {
        "mmlu": "mmlu",
        "truthfulqa": "truthfulqa",
        "truthfulqagen": "truthfulqa",
        "gpqa": "gpqa",
        "gpqadiamond": "gpqa",
        "popqa": "popqa",
        "pop_qa": "popqa",
        "gsm8k": "gsm8k",
        "trait": "personality_trait",
        "personalitytrait": "personality_trait",
        "bfi": "personality_bfi",
        "personalitybfi": "personality_bfi",
    }
    return aliases.get(normalized, normalized)


def _normalize_trait_name(name: str) -> str:
    normalized = "".join(ch for ch in name.lower() if ch.isalnum())
    aliases = {
        "open": "Openness",
        "openness": "Openness",
        "conscientiousness": "Conscientiousness",
        "extraversion": "Extraversion",
        "agreeableness": "Agreeableness",
        "neuroticism": "Neuroticism",
        "machiavellianism": "Machiavellianism",
        "narcissism": "Narcissism",
        "psychopathy": "Psychopathy",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unknown TRAIT split '{name}'. "
            f"Valid values: {sorted(set(aliases.values()))}"
        )
    return aliases[normalized]


def _filter_trait_task(task: Task, trait_name: str) -> Task:
    dataset = getattr(task, "dataset", None)
    samples = getattr(dataset, "samples", None)
    if not isinstance(dataset, MemoryDataset) or samples is None:
        raise TypeError(
            "personality_trait filtering expects a MemoryDataset-backed task"
        )

    canonical_trait = _normalize_trait_name(trait_name)
    filtered_samples = [
        sample
        for sample in samples
        if (sample.metadata or {}).get("trait") == canonical_trait
    ]
    if not filtered_samples:
        raise ValueError(
            f"No TRAIT samples matched requested trait '{canonical_trait}'"
        )

    task.dataset = MemoryDataset(
        samples=filtered_samples,
        name=f"{getattr(dataset, 'name', 'personality_trait')}_{canonical_trait}",
    )
    return task


def _build_single_trait_task(*, personality: str, trait: str) -> Task:
    from inspect_evals.personality.personality import (
        create_task,
        enrich_dataset,
        record_to_sample_TRAIT,
    )
    from inspect_evals.personality.prompts.system import get_system_prompt
    from inspect_evals.utils.huggingface import hf_dataset

    canonical_trait = _normalize_trait_name(trait)
    dataset = hf_dataset(
        path="mirlab/TRAIT",
        split=canonical_trait,
        sample_fields=record_to_sample_TRAIT,
        cached=False,
        token=os.getenv("HF_TOKEN"),
    )
    dataset = enrich_dataset(dataset, {"answer_mapping": {"A": 1, "B": 1, "C": 0, "D": 0}})
    system_msg = get_system_prompt("trait", personality)
    return create_task(dataset, system_msg)


def build_benchmark_task(spec: InspectBenchmarkSpec) -> Task:
    """Build an Inspect task for a known benchmark."""
    benchmark = _canonical_name(spec.benchmark)
    kwargs: dict[str, Any] = dict(spec.benchmark_args)

    if benchmark == "mmlu":
        from inspect_evals.mmlu.mmlu import mmlu_0_shot

        return mmlu_0_shot(**kwargs)

    if benchmark == "truthfulqa":
        from inspect_evals.truthfulqa import truthfulqa

        return truthfulqa(**kwargs)

    if benchmark == "gpqa":
        from inspect_evals.gpqa.gpqa import gpqa_diamond

        return gpqa_diamond(**kwargs)

    if benchmark == "gsm8k":
        from inspect_evals.gsm8k import gsm8k

        return gsm8k(**kwargs)

    if benchmark == "personality_trait":
        from inspect_evals.personality.personality import personality_TRAIT

        trait_name = kwargs.pop("trait", None)
        if trait_name is not None:
            personality = str(kwargs.pop("personality", ""))
            if kwargs:
                task = personality_TRAIT(personality=personality, **kwargs)
                return _filter_trait_task(task, str(trait_name))
            return _build_single_trait_task(
                personality=personality,
                trait=str(trait_name),
            )

        return personality_TRAIT(**kwargs)

    if benchmark == "personality_bfi":
        from inspect_evals.personality.personality import personality_BFI

        return personality_BFI(**kwargs)

    if benchmark == "popqa":
        limit = spec.limit if spec.limit is not None else kwargs.pop("limit", None)
        return _build_popqa_task(limit=limit)

    raise ValueError(
        f"Unknown benchmark '{spec.benchmark}'. "
        "Supported benchmarks: mmlu, truthfulqa, gpqa, popqa, gsm8k, "
        "personality_trait, personality_bfi"
    )
