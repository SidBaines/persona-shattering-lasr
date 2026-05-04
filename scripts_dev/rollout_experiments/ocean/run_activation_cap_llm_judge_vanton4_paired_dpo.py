#!/usr/bin/env python3
"""Activation-capping LLM-judge sweep for OCEAN± vanton4_paired_dpo (own-trait).

For each of the 10 OCEAN± vanton4_paired_dpo adapters, runs a sweep over
activation-capping fractions on the persona's axis, generates rollouts on
the trait-specific open-ended question set (data/ocean_open_ended/<trait>.jsonl),
and scores each response with the trait's registered LLM judge metric (e.g.
``agreeableness_v2``). Results upload to::

    fine_tuning/llama-3.1-8b-it/ocean/<trait>/<direction>/vanton4_paired_dpo/
        evals/llm_judge_activation_capping_sweep/<persona_slug>/

Mirrors the LoRA-scale judge sweep at ``…/evals/llm_judge_lora_scale_sweep/``,
but with ActivationCapProvider instead of VLLMLoRaScaleProvider. Rollouts are
HF-transformers-based (vLLM cannot host the per-layer forward hooks that
activation capping uses), so this is ~5–10× slower than the LoRA judge sweep.

The capping axis files for vanton4_paired_dpo OCEAN± live next to each LoRA at
``<lora_parent>/activation_capping/<persona_slug>_axis.pt`` (the layout
``compute_axis.py`` writes them to). The OCEAN_REGISTRY currently has
``axis_slug=None`` for these entries because the registry's ``axis_slug``
helper assumes the legacy top-level layout (``activation_capping/<slug>/...``).
We bypass ``axis_slug`` and derive paths from the LoRA path directly, the way
the MCQ ``activation_capping`` configs already do.

Fractions: 5 evenly spaced points ``[-2.0, -1.0, 0.0, 1.0, 2.0]``, matching
the LoRA-judge sweep grid (``_shared.SCALE_POINTS``) and the MCQ activation-
capping x-axis range.

Usage::

    # Run all 10 OCEAN± rows:
    uv run python scripts_dev/rollout_experiments/ocean/run_activation_cap_llm_judge_vanton4_paired_dpo.py

    # Run a single row (useful for testing or sharding by GPU):
    uv run python scripts_dev/rollout_experiments/ocean/run_activation_cap_llm_judge_vanton4_paired_dpo.py --slug n_minus
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import random

import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

from dotenv import load_dotenv

from src_dev.common.lora_catalogue import HF_REPO, OCEAN_REGISTRY
from src_dev.rollout_generation.model_providers import (
    ActivationCapProvider,
    _resolve_hf_path,
)
from src_dev.sweep import (
    ExperimentConfig,
    OutputPathConfig,
    SweepConfig,
    run_sweep,
    single_turn_conditions,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
BASE_MODEL_SLUG = "llama-3.1-8b-it"

# 5 fractions, same spacing as the LoRA-judge sweep
# (scripts_dev/evals/llm_judge_sweep/configs/vanton4_paired_dpo/_shared.py
# SCALE_POINTS) and same x-axis range as the MCQ activation_capping configs.
FRACTIONS = [-2.0, -1.0, 0.0, 1.0, 2.0]

# Result directory under each adapter's evals/ prefix.
SWEEP_NAME = "llm_judge_activation_capping_sweep"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _axis_uris_from_lora(adapter_path_in_repo: str, persona_slug: str) -> tuple[str, str]:
    """Build hf:// URIs for the axis + per-layer-range files next to the LoRA.

    OCEAN± vanton4_paired_dpo adapter paths look like
    ``fine_tuning/.../vanton4_paired_dpo/lora/<adapter_name>``. We walk up two
    levels to ``<...>/vanton4_paired_dpo`` and append
    ``activation_capping/<slug>_axis.pt``.
    """
    p = Path(adapter_path_in_repo)
    parent = p.parent.parent if p.parent.name == "lora" else p
    axis = f"hf://{HF_REPO}/{parent.as_posix()}/activation_capping/{persona_slug}_axis.pt"
    per_layer = f"hf://{HF_REPO}/{parent.as_posix()}/activation_capping/{persona_slug}_per_layer_range.pt"
    return axis, per_layer


def _read_capping_layers(axis_uri: str) -> list[int]:
    """Load recommended capping layers from axis metadata.

    Falls back to layers 17–31 (upper half of Llama-3.1-8B's 32 layers) if
    the metadata key is missing — mirrors the fallback in
    ``scripts_dev/rollout_experiments/ocean/generate_rollouts.py``.
    """
    import torch

    local = _resolve_hf_path(axis_uri)
    data = torch.load(local, map_location="cpu", weights_only=False)
    layers = data.get("metadata", {}).get("recommended_capping_layers")
    if layers:
        print(f"  Using recommended capping layers from axis metadata: {list(layers)}")
        return list(layers)
    print("  WARN: recommended_capping_layers not in axis metadata, "
          "falling back to layers 17-31")
    return list(range(17, 32))


# ─────────────────────────────────────────────────────────────────────────────
# Per-trait sweep
# ─────────────────────────────────────────────────────────────────────────────


def run_one(slug: str, *, max_samples: int, num_rollouts: int) -> None:
    trait_def = OCEAN_REGISTRY[slug]
    if trait_def.eval_metric is None:
        print(f"[skip] {slug}: no eval_metric configured in OCEAN_REGISTRY.")
        return

    dataset_path = f"data/ocean_open_ended/{trait_def.trait_name}.jsonl"
    if not Path(dataset_path).exists():
        print(f"[skip] {slug}: dataset {dataset_path} not found.")
        return

    axis_uri, per_layer_uri = _axis_uris_from_lora(
        trait_def.adapter_path_in_repo, persona_slug=slug,
    )

    print()
    print("=" * 70)
    print(f"  {slug} — activation-capping LLM judge sweep")
    print(f"  Adapter:   {trait_def.adapter_ref}")
    print(f"  Axis:      {axis_uri}")
    print(f"  PerLayer:  {per_layer_uri}")
    print(f"  Dataset:   {dataset_path}")
    print(f"  Metric:    {trait_def.eval_metric}")
    print(f"  Fractions: {FRACTIONS}")
    print("=" * 70)

    capping_layers = _read_capping_layers(axis_uri)

    provider = ActivationCapProvider(
        base_model=BASE_MODEL,
        axis_path=axis_uri,
        per_layer_range_path=per_layer_uri,
        fractions=FRACTIONS,
        capping_layers=capping_layers,
    )

    experiment = ExperimentConfig(
        assistant_model=BASE_MODEL,
        assistant_provider="local",  # HF transformers — required for capping hooks
        assistant_temperature=1.0,
        assistant_top_p=1.0,
        assistant_max_new_tokens=2048,
        assistant_batch_size=32,
        user_model="z-ai/glm-4.5-air:free",
        user_provider="openrouter",
        user_temperature=0.7,
        user_top_p=1.0,
        user_max_new_tokens=4096,
        user_batch_size=32,
        user_max_concurrent=32,
        dataset_path=dataset_path,
        max_samples=max_samples,
        dataset_seed=SEED,
        num_rollouts=num_rollouts,
        turns_per_phase=[1],
    )

    # Single-turn, no system-prompt sweep — same as the LoRA judge does.
    conditions = single_turn_conditions({"no_prompt": None})

    output = OutputPathConfig(
        scratch_root=Path("scratch/monorepo"),
        hf_repo=HF_REPO,
        base_model=BASE_MODEL_SLUG,
        category="ocean",
        trait=trait_def.output_trait_path,  # "<trait_name>/<direction>"
        training_run=trait_def.version,     # "vanton4_paired_dpo"
        stage_dir="evals",
        eval_name=f"{SWEEP_NAME}/{slug}",
    )

    sweep_config = SweepConfig(
        provider=provider,
        conditions=conditions,
        evaluations=[trait_def.eval_metric],
        experiment=experiment,
        output=output,
        skip_completed=True,
        on_cell_error="warn",
        max_concurrent_conditions=1,
    )

    print(f"  Output:    {output.scratch_dir}")
    print(f"  HF path:   {output.hf_path}")
    output_root = run_sweep(sweep_config)
    print(f"  Done. Results in {output_root}/")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--slug",
        choices=sorted(OCEAN_REGISTRY.keys()),
        help="Single OCEAN± slug to run (e.g. n_minus). Default: all 10.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=240,
        help="Max prompts per dataset (default: 240, matches the LoRA judge sweep).",
    )
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=1,
        help="Rollouts per prompt (default: 1, matches the LoRA judge sweep).",
    )
    args = parser.parse_args()

    load_dotenv()

    slugs = [args.slug] if args.slug else list(OCEAN_REGISTRY.keys())

    for slug in slugs:
        try:
            run_one(slug, max_samples=args.max_samples, num_rollouts=args.num_rollouts)
        except Exception as exc:  # noqa: BLE001 — soft-fail per row
            print(f"[FAIL] {slug}: {type(exc).__name__}: {exc}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
