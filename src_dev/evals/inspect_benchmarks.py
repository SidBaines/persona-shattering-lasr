"""Benchmark task builders for Inspect-based eval runs."""

from __future__ import annotations

import json
from typing import Any

from datasets import load_dataset
from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import includes
from inspect_ai.solver import generate

from src_dev.evals.config import InspectBenchmarkSpec


TRAIT_SAMPLE_SPLITS = (
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism",
    "Machiavellianism",
    "Narcissism",
    "Psychopathy",
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


def _load_trait_dataset(
    samples_per_trait: int = 25,
    trait_splits: list[str] | tuple[str, ...] | None = None,
    shuffle_choices: bool = True,
    seed: int = 42,
) -> MemoryDataset:
    """Load and sample TRAIT dataset evenly across selected trait splits.

    Returns an enriched MemoryDataset with answer_mapping metadata attached
    to every sample.  Shared by both the text-based and logprob-based task
    builders.

    Args:
        samples_per_trait: Number of questions per trait split.
        trait_splits: Which TRAIT splits to include (default: all 8).
        shuffle_choices: If True, randomly shuffle each sample's choices and
            remap the per-sample answer_mapping to match.  This removes
            positional bias (TRAIT always puts high-trait answers at A/B).
        seed: Random seed for choice shuffling.
    """
    import os
    from inspect_evals.personality.personality import (
        enrich_dataset,
        hf_dataset,
        record_to_sample_TRAIT,
    )

    if trait_splits is None:
        splits = list(TRAIT_SAMPLE_SPLITS)
    else:
        splits = []
        allowed = set(TRAIT_SAMPLE_SPLITS)
        for split in trait_splits:
            if split not in allowed:
                raise ValueError(
                    f"Unknown TRAIT split '{split}'. Valid options: {', '.join(TRAIT_SAMPLE_SPLITS)}"
                )
            splits.append(split)
        if not splits:
            raise ValueError("trait_splits must contain at least one TRAIT split")

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
    original_answer_mapping: dict[str, int] = {"A": 1, "B": 1, "C": 0, "D": 0}
    meta = {"answer_mapping": original_answer_mapping}
    combined_ds = enrich_dataset(combined_ds, meta)

    if shuffle_choices:
        # Capture each sample's original target before shuffling so we can
        # remap answer_mapping afterwards.
        orig_targets = [list(s.target) if isinstance(s.target, list) else [s.target]
                        for s in combined_ds]
        combined_ds.shuffle_choices(seed=seed)
        for sample, orig_target in zip(combined_ds, orig_targets):
            new_target = sample.target if isinstance(sample.target, list) else [sample.target]
            # orig_target[i] = old letter at position i
            # new_target[i]  = new letter that the original position i moved to
            sample.metadata["answer_mapping"] = {
                new_letter: original_answer_mapping[old_letter]
                for old_letter, new_letter in zip(orig_target, new_target)
            }

    return combined_ds


def _build_trait_sampled_task(
    samples_per_trait: int = 25,
    trait_splits: list[str] | tuple[str, ...] | None = None,
    max_tokens: int | None = 32,
) -> Task:
    """Build a TRAIT task sampling evenly across selected trait splits.

    The standard personality_TRAIT task concatenates all splits then applies
    a global limit, so a small limit yields only the first trait (Openness).
    This builder caps each split independently before combining.

    max_tokens defaults to 32 — TRAIT is MCQ so only a letter is needed.
    """
    from inspect_evals.personality.personality import (
        create_task,
        get_system_prompt,
    )

    combined_ds = _load_trait_dataset(samples_per_trait, trait_splits)
    system_msg = get_system_prompt("trait", "")
    task = create_task(combined_ds, system_msg)
    if max_tokens is not None:
        from inspect_ai.model import GenerateConfig
        task.config = task.config.merge(GenerateConfig(max_tokens=max_tokens))
    return task


def _build_trait_logprobs_task(
    samples_per_trait: int = 25,
    trait_splits: list[str] | tuple[str, ...] | None = None,
    prefill: str = "ANSWER: ",
) -> Task:
    """Build a TRAIT task that uses logprob-based scoring.

    Same dataset and prompt formatting as _build_trait_sampled_task, but
    replaces the text-based multiple_choice solver + any_choice scorer with
    a logprob-based solver and scorer that reads the model's probability
    distribution over choice tokens.

    Args:
        samples_per_trait: Number of questions per trait split.
        trait_splits: Which TRAIT splits to include (default: all 8).
        prefill: Forced assistant prefill before generation. Set to ""
            to disable.
    """
    from inspect_ai.model import GenerateConfig
    from inspect_ai.solver import system_message

    from inspect_evals.personality.personality import get_system_prompt

    from src_dev.evals.personality.logprob_scorer import (
        logprob_multiple_choice,
        logprob_trait_ratio,
        logprob_trait_scorer,
    )

    combined_ds = _load_trait_dataset(samples_per_trait, trait_splits)
    system_msg = get_system_prompt("trait", "")

    task = Task(
        dataset=combined_ds,
        solver=[
            system_message(system_msg),
            logprob_multiple_choice(prefill=prefill),
        ],
        scorer=logprob_trait_scorer(),
        metrics=[logprob_trait_ratio()],
        config=GenerateConfig(
            logprobs=True,
            top_logprobs=20,
            max_tokens=1,
        ),
    )
    return task


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
        "personalitytraitlogprobs": "personality_trait_logprobs",
        "traitlogprobs": "personality_trait_logprobs",
    }
    return aliases.get(normalized, normalized)


def build_benchmark_task(spec: InspectBenchmarkSpec) -> Task:
    """Build an Inspect task for a known benchmark."""
    benchmark = _canonical_name(spec.benchmark)
    kwargs: dict[str, Any] = dict(spec.benchmark_args)

    if benchmark == "mmlu":
        import random
        from collections import defaultdict

        from inspect_evals.mmlu.mmlu import mmlu_0_shot
        from inspect_evals.utils import filter_duplicate_ids

        max_samples = kwargs.pop("max_samples", None)
        task = mmlu_0_shot(**kwargs)
        task.dataset = filter_duplicate_ids(task.dataset)

        if max_samples is not None:
            # Stratified sampling: distribute max_samples evenly across subjects,
            # with remainder allocated round-robin alphabetically.
            by_subject: dict[str, list] = defaultdict(list)
            for sample in task.dataset:
                by_subject[sample.metadata["subject"]].append(sample)
            subjects = sorted(by_subject)
            per_subject, remainder = divmod(int(max_samples), len(subjects))
            sampled: list = []
            for i, subj in enumerate(subjects):
                n = per_subject + (1 if i < remainder else 0)
                pool = by_subject[subj]
                sampled.extend(random.sample(pool, min(n, len(pool))))
            random.shuffle(sampled)
            task.dataset = MemoryDataset(sampled)

        # The task hardcodes temperature=0.0, which the HF local backend rejects
        # (it requires do_sample=False for greedy decoding instead).  Clear it so
        # Inspect does not forward temperature to the backend.
        task.config.temperature = None
        # MCQ only needs a single letter answer — cap generation to avoid
        # running to the provider's default max_tokens (2048) per sample.
        task.config.max_tokens = 32
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
        trait_splits = kwargs.pop("trait_splits", None)
        max_tokens = kwargs.pop("max_tokens", None)
        return _build_trait_sampled_task(
            samples_per_trait=samples_per_trait,
            trait_splits=trait_splits,
            max_tokens=int(max_tokens) if max_tokens is not None else None,
        )

    if benchmark == "personality_trait_logprobs":
        samples_per_trait = int(kwargs.pop("samples_per_trait", 25))
        trait_splits = kwargs.pop("trait_splits", None)
        prefill = kwargs.pop("prefill", "ANSWER: ")
        return _build_trait_logprobs_task(
            samples_per_trait=samples_per_trait,
            trait_splits=trait_splits,
            prefill=str(prefill),
        )

    raise ValueError(
        f"Unknown benchmark '{spec.benchmark}'. "
        "Supported benchmarks: mmlu, truthfulqa, gpqa, popqa, gsm8k, "
        "personality_bfi, personality_trait, personality_trait_sampled, personality_trait_logprobs"
    )
