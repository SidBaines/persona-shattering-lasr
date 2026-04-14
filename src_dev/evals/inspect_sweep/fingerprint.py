"""Content-addressed fingerprint for a general Inspect-benchmark cell sweep.

Hashes every config field that changes what the model sees or how it is
scored across an arbitrary set of Inspect benchmarks (MMLU, GPQA,
TruthfulQA, etc.). Adapters are NOT hashed (they live in the cell
identity), and throughput-only fields like ``BATCH_SIZE`` are NOT hashed.

The benchmark set itself IS hashed (as a canonicalised list of
``(name, benchmark, benchmark_args, limit, n_runs)`` tuples) so swapping
or re-parameterising any benchmark forks the cache.
"""

from __future__ import annotations

from typing import Any, Sequence

from src_dev.evals.cell_sweep.fingerprint import fingerprint_from_fields


def _canonicalise_benchmark_spec(spec: Any) -> dict[str, Any]:
    return {
        "name": str(spec.name),
        "benchmark": str(spec.benchmark),
        "benchmark_args": dict(spec.benchmark_args or {}),
        "limit": spec.limit,
        "n_runs": int(getattr(spec, "n_runs", 1) or 1),
    }


def inspect_sweep_fingerprint(
    *,
    base_model: str,
    benchmark_specs: Sequence[Any],
    seed: int,
    temperature: float,
    length: int = 10,
) -> str:
    """Compute the inspect-sweep rollout fingerprint.

    ``benchmark_specs`` may be :class:`InspectBenchmarkSpec` instances or any
    object with matching attributes. Order is preserved in the hash — two
    sweeps with the same benchmarks in different orders collide only if
    the order is identical. Callers that want order-insensitive caching
    should sort their specs by ``name`` before fingerprinting.
    """
    fields: dict[str, Any] = {
        "base_model": base_model,
        "benchmarks": [_canonicalise_benchmark_spec(s) for s in benchmark_specs],
        "seed": int(seed),
        "temperature": float(temperature),
    }
    return fingerprint_from_fields(fields, length=length)
