"""Benchmark task builders for Inspect-based eval runs."""

from __future__ import annotations

import json
from typing import Any, Literal

from datasets import load_dataset
from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from inspect_ai.scorer import includes
from inspect_ai.solver import generate

from scripts.evals.config import InspectBenchmarkSpec


def inject_system_prompt(
    task: Task,
    system_prompt: str,
    mode: Literal["prepend", "replace"] = "prepend",
) -> Task:
    """Return a new Task with *system_prompt* injected into every sample's input.

    Args:
        task: The Inspect Task whose dataset samples will be modified.
        system_prompt: Text for the injected system message.
        mode: How to handle samples that already contain a system message.
            ``"prepend"`` inserts the custom prompt *before* any existing system
            message, preserving the benchmark's own framing.
            ``"replace"`` discards any existing system message.

    Returns:
        A new Task with the same config but a modified dataset.
    """
    new_samples: list[Sample] = []
    for sample in task.dataset:
        input_ = sample.input
        if isinstance(input_, str):
            existing_sys: list = []
            non_sys: list = [ChatMessageUser(content=input_)]
        else:
            messages_list = list(input_)
            existing_sys = [m for m in messages_list if isinstance(m, ChatMessageSystem)]
            non_sys = [m for m in messages_list if not isinstance(m, ChatMessageSystem)]

        new_sys = ChatMessageSystem(content=system_prompt)
        if mode == "replace" or not existing_sys:
            rebuilt = [new_sys] + non_sys
        else:  # prepend
            rebuilt = [new_sys] + existing_sys + non_sys

        new_samples.append(sample.model_copy(update={"input": rebuilt}))

    return Task(
        dataset=MemoryDataset(new_samples, name=task.dataset.name),
        setup=task.setup,
        solver=task.solver,
        cleanup=task.cleanup,
        scorer=task.scorer,
        metrics=task.metrics,
        config=task.config,
        sandbox=task.sandbox,
        epochs=task.epochs,
        fail_on_error=task.fail_on_error,
        name=task.name,
        metadata=task.metadata,
    )


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


def _build_trait_sampled_task(samples_per_trait: int = 25) -> Task:
    """Build a TRAIT task sampling evenly across all 8 trait splits.

    The standard personality_TRAIT task concatenates all splits then applies
    a global limit, so a small limit yields only the first trait (Openness).
    This builder caps each split independently before combining.
    """
    import os
    from inspect_evals.personality.personality import (
        create_task,
        enrich_dataset,
        get_system_prompt,
        hf_dataset,
        record_to_sample_TRAIT,
    )

    splits = [
        "Openness", "Conscientiousness", "Extraversion", "Agreeableness",
        "Neuroticism", "Machiavellianism", "Narcissism", "Psychopathy",
    ]
    all_samples: list[Sample] = []
    for split in splits:
        split_samples = list(hf_dataset(
            path="mirlab/TRAIT",
            split=split,
            sample_fields=record_to_sample_TRAIT,
            cached=False,
            token=os.getenv("HF_TOKEN"),
        ))
        all_samples.extend(split_samples[:samples_per_trait])

    combined_ds = MemoryDataset(all_samples)
    meta = {"answer_mapping": {"A": 1, "B": 1, "C": 0, "D": 0}}
    combined_ds = enrich_dataset(combined_ds, meta)
    system_msg = get_system_prompt("trait", "")
    return create_task(combined_ds, system_msg)


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
        "personalitybfi": "personality_bfi",
        "bfi": "personality_bfi",
        "personalitytrait": "personality_trait",
        "trait": "personality_trait",
        "personalitytraitsampled": "personality_trait_sampled",
        "traitsampled": "personality_trait_sampled",
    }
    return aliases.get(normalized, normalized)


def build_benchmark_task(spec: InspectBenchmarkSpec) -> Task:
    """Build an Inspect task for a known benchmark."""
    benchmark = _canonical_name(spec.benchmark)
    kwargs: dict[str, Any] = dict(spec.benchmark_args)

    if benchmark == "mmlu":
        from inspect_evals.mmlu.mmlu import mmlu_0_shot
        from inspect_evals.utils import filter_duplicate_ids

        task = mmlu_0_shot(**kwargs)
        task.dataset = filter_duplicate_ids(task.dataset)
        # The task hardcodes temperature=0.0, which the HF local backend rejects
        # (it requires do_sample=False for greedy decoding instead).  Clear it so
        # Inspect does not forward temperature to the backend.
        task.config.temperature = None
        return task

    if benchmark == "truthfulqa":
        from inspect_evals.truthfulqa import truthfulqa

        return truthfulqa(**kwargs)

    if benchmark == "gpqa":
        from inspect_evals.gpqa.gpqa import gpqa_diamond

        return gpqa_diamond(**kwargs)

    if benchmark == "gsm8k":
        from inspect_evals.gsm8k import gsm8k

        return gsm8k(**kwargs)

    if benchmark == "popqa":
        limit = spec.limit if spec.limit is not None else kwargs.pop("limit", None)
        return _build_popqa_task(limit=limit)

    if benchmark == "personality_bfi":
        from inspect_evals.personality import personality_BFI

        return personality_BFI(**kwargs)

    if benchmark == "personality_trait":
        from inspect_evals.personality import personality_TRAIT

        return personality_TRAIT(**kwargs)

    if benchmark == "personality_trait_sampled":
        # Use benchmark_args["samples_per_trait"] rather than spec.limit, because
        # spec.limit is also passed as a global Inspect limit by the runner and
        # would re-truncate the already-sampled combined dataset back to one trait.
        samples_per_trait = int(kwargs.pop("samples_per_trait", 25))
        return _build_trait_sampled_task(samples_per_trait=samples_per_trait)

    raise ValueError(
        f"Unknown benchmark '{spec.benchmark}'. "
        "Supported benchmarks: mmlu, truthfulqa, gpqa, popqa, gsm8k, personality_bfi, personality_trait, personality_trait_sampled"
    )
