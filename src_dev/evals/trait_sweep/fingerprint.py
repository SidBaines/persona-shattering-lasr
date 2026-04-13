"""Content-addressed fingerprint for a TRAIT-benchmark cell sweep.

The fingerprint hashes every config field that changes what the model sees
or how it is scored in a TRAIT run — benchmark kind, dataset slice, shuffle
behaviour, logprob-scoring knobs, generation temperature. Adapters are NOT
hashed (they live in the cell identity), and throughput-only fields like
``BATCH_SIZE`` are NOT hashed. ``trait_splits`` is also NOT hashed — each
trait's outputs live in their own per-trait subdir within the cell, so
running a subset of traits simply leaves other traits absent (not cached
under a different fingerprint).
"""

from __future__ import annotations

from typing import Any

from src_dev.evals.cell_sweep.fingerprint import fingerprint_from_fields


def trait_fingerprint(
    *,
    base_model: str,
    benchmark: str,
    samples_per_trait: int,
    shuffle_choices: bool,
    seed: int,
    temperature: float,
    prefill: str,
    min_choice_mass: float,
    dynamic_mass_filter: bool,
    template: str | None,
    max_tokens: int | None,
    length: int = 10,
) -> str:
    """Compute the TRAIT-sweep rollout fingerprint.

    Fields are canonicalised before hashing so equivalent configs collide;
    ``None`` values are kept as JSON null.
    """
    fields: dict[str, Any] = {
        "base_model": base_model,
        "benchmark": benchmark,
        "samples_per_trait": int(samples_per_trait),
        "shuffle_choices": bool(shuffle_choices),
        "seed": int(seed),
        "temperature": float(temperature),
        "prefill": prefill,
        "min_choice_mass": float(min_choice_mass),
        "dynamic_mass_filter": bool(dynamic_mass_filter),
        "template": template,
        "max_tokens": max_tokens,
    }
    return fingerprint_from_fields(fields, length=length)
