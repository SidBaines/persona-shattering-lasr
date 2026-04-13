#!/usr/bin/env python3
"""Cell-oriented bloom sweep runner (baseline / single-adapter / combo).

Replaces the stage-chained flow of legacy :mod:`runner` with cell-level
content-addressed caching on HuggingFace. Two cache layers:

1. **Trait-scoped ideation cache** (manually versioned) at
   ``evals/bloom_ideation/{trait}/v{N}/{ideation_fp}/`` — one per trait, shared
   across every ``(model, adapter, combo)`` run.
2. **Per-cell rollout + judgment** at
   :meth:`CanonicalCell.hf_dir(model_slug, "bloom_{trait}", rollout_cell_fp)` —
   rollouts + per-quality judge outputs.

Key properties:

- Changing the judge model adds a new ``judge_runs/{judge}/`` subtree without
  invalidating rollouts.
- Bumping ``SCENARIO_VERSION`` forces every downstream rollout cell to be
  recomputed (the version is hashed into ``rollout_cell_fp``). Old
  ``v{N}/`` artifacts remain on HF.
- Bloom writes to ``bloom-results/{behavior}/`` which is shared across cells,
  so cells run sequentially (enforced by the orchestration loop).

Config compatibility:
- Combo configs: ``ADAPTERS: list[AdapterSpec]`` + ``SCALES_PER_ADAPTER: dict``.
- Legacy single-adapter configs (``ADAPTER_REF`` + ``SCALE_POINTS``) are
  auto-promoted to the combo shape on load.
- Baseline-only configs: ``ADAPTERS=[]`` produces a single baseline cell.
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from dotenv import load_dotenv

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

load_dotenv()

from src_dev.evals.bloom.bloom_runtime import (
    STAGES,
    bake_cells_for_bloom,
    build_trait_context,
    clean_bloom_results_dir,
    ensure_vllm_running,
    load_bloom_config,
    patched_bloom_data,
    prompts_subset,
    route_bloom_outputs,
    run_bloom_stage,
    seed_ideation_into_bloom_results,
    validate_evaluator_models,
)
from src_dev.evals.bloom.defaults import check_sweep_defaults, confirm_or_abort
from src_dev.evals.bloom.fingerprint import (
    ideation_fingerprint,
    rollout_cell_fingerprint,
)
from src_dev.evals.bloom.ideation_cache import (
    IDEATION_FILENAME,
    IDEATION_META_FILENAME,
    UNDERSTANDING_FILENAME,
    hydrate_ideation,
    ideation_hf_dir,
    upload_ideation,
)
from src_dev.evals.bloom.judgment_postprocess import split_judgment_into_qualities
from src_dev.evals.bloom.layout import (
    CELL_INFO_RELPATH,
    IDEATION_REF_RELPATH,
    UPLOAD_ALLOW_PATTERNS,
    hydrate_cell,
    upload_cell,
)
from src_dev.evals.cell_sweep.cell_identity import AdapterSpec, CanonicalCell
from src_dev.rollout_generation.model_providers import cleanup_baked_dir
from src_dev.utils.hf_hub import login_from_env

HF_REPO_ID = "persona-shattering-lasr/monorepo"
EVAL_NAME_BASE = "bloom"
SCRATCH_ROOT = Path("scratch/monorepo")
BAKED_ROOT = Path("scratch/baked_bloom_adapters")
BLOOM_WORK_ROOT = Path("scratch/bloom_work")

DEFAULT_QUALITY_KEYS: tuple[str, ...] = ("behavior_presence", "coherence")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_flags() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cell-oriented bloom sweep runner.")
    p.add_argument("--config", required=True,
                   help="Python module path to the config constants.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-upload", action="store_true")
    p.add_argument("--skip-rollouts", action="store_true",
                   help="Skip rollout stage (use cached rollouts only).")
    p.add_argument("--skip-judge", action="store_true",
                   help="Skip judgment stage (use cached judge outputs only).")
    p.add_argument("--allow-custom-fingerprint", action="store_true",
                   help="Skip the canonical-defaults drift prompt.")
    p.add_argument("--no-vllm", action="store_true",
                   help="Disable automatic vLLM launch.")
    p.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES,
                   help="Subset of stages to run. Stages outside this set are "
                        "never re-run even if their inputs are missing.")
    p.add_argument("--judgment-models", nargs="+", default=None,
                   help="Judge model short names (override JUDGMENT_MODELS from "
                        "config).")
    return p.parse_args()


def _load_config_module(module_path: str) -> ModuleType:
    return importlib.import_module(module_path)


# ---------------------------------------------------------------------------
# Config normalisation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NormalisedConfig:
    adapters: tuple[AdapterSpec, ...]
    scales_per_adapter: dict[str, tuple[float, ...]]
    traits: tuple[str, ...]
    base_model: str
    base_model_slug: str
    eval_name: str
    bloom_data_dir: Path
    judgment_models: tuple[str, ...]
    scenario_version: int
    num_scenarios: int
    num_reps: int
    max_turns: int
    rollout_max_tokens: int
    modality: str
    no_user_mode: bool
    anonymous_target: bool
    temperature: float
    judge_temperature: float
    ideation_temperature: float
    evaluator_reasoning_effort: str
    ideation_reasoning_effort: str
    target_reasoning_effort: str
    understanding_model: str
    understanding_max_tokens: int
    ideation_model: str
    ideation_max_tokens: int
    rollout_evaluator_model: str
    web_search: bool
    variation_dimensions: tuple[str, ...] | None
    selected_variations: tuple[str, ...] | None
    quality_keys: tuple[str, ...]
    seed: int
    max_lora_rank: int
    hf_repo: str


def _normalise_config(
    cfg: ModuleType,
    *,
    judgment_models_override: list[str] | None,
) -> NormalisedConfig:
    if hasattr(cfg, "ADAPTERS"):
        adapters = tuple(cfg.ADAPTERS)
        if adapters:
            scales_per_adapter = {
                a.slug: tuple(float(s) for s in cfg.SCALES_PER_ADAPTER[a.slug])
                for a in adapters
            }
        else:
            scales_per_adapter = {}
    elif hasattr(cfg, "ADAPTER_REF"):
        spec = AdapterSpec.from_ref(cfg.ADAPTER_REF)
        adapters = (spec,)
        scales_per_adapter = {
            spec.slug: tuple(float(s) for s in cfg.SCALE_POINTS)
        }
    else:
        adapters = ()
        scales_per_adapter = {}

    if hasattr(cfg, "TRAITS"):
        traits = tuple(cfg.TRAITS)
    elif hasattr(cfg, "TRAIT"):
        traits = (cfg.TRAIT,)
    else:
        raise ValueError("Config must set TRAITS (list) or TRAIT (single).")

    judgment_models = tuple(
        judgment_models_override
        if judgment_models_override is not None
        else getattr(cfg, "JUDGMENT_MODELS", None) or []
    )
    if not judgment_models:
        raise ValueError(
            "Config must set JUDGMENT_MODELS (list of judge short names); "
            "or pass --judgment-models on the CLI."
        )

    variation_dimensions_raw = getattr(cfg, "VARIATION_DIMENSIONS", None)
    selected_variations_raw = getattr(cfg, "SELECTED_VARIATIONS", None)

    return NormalisedConfig(
        adapters=adapters,
        scales_per_adapter=scales_per_adapter,
        traits=traits,
        base_model=cfg.BASE_MODEL,
        base_model_slug=cfg.BASE_MODEL_SLUG,
        eval_name=getattr(cfg, "EVAL_NAME_CANONICAL", EVAL_NAME_BASE),
        bloom_data_dir=Path(getattr(cfg, "BLOOM_DATA_DIR", "bloom-data")).resolve(),
        judgment_models=judgment_models,
        scenario_version=int(getattr(cfg, "SCENARIO_VERSION", 1)),
        num_scenarios=int(cfg.NUM_SCENARIOS),
        num_reps=int(cfg.NUM_REPS),
        max_turns=int(cfg.MAX_TURNS),
        rollout_max_tokens=int(cfg.ROLLOUT_MAX_TOKENS),
        modality=getattr(cfg, "MODALITY", "simenv"),
        no_user_mode=bool(getattr(cfg, "NO_USER_MODE", False)),
        anonymous_target=bool(getattr(cfg, "ANONYMOUS_TARGET", False)),
        temperature=float(getattr(cfg, "TEMPERATURE", 1.0)),
        judge_temperature=float(
            getattr(cfg, "JUDGE_TEMPERATURE", getattr(cfg, "TEMPERATURE", 1.0))
        ),
        ideation_temperature=float(
            getattr(cfg, "IDEATION_TEMPERATURE", getattr(cfg, "TEMPERATURE", 1.0))
        ),
        evaluator_reasoning_effort=getattr(cfg, "EVALUATOR_REASONING_EFFORT", "low"),
        ideation_reasoning_effort=getattr(
            cfg, "IDEATION_REASONING_EFFORT",
            getattr(cfg, "EVALUATOR_REASONING_EFFORT", "low"),
        ),
        target_reasoning_effort=getattr(cfg, "TARGET_REASONING_EFFORT", "medium"),
        understanding_model=cfg.UNDERSTANDING_MODEL,
        understanding_max_tokens=int(cfg.UNDERSTANDING_MAX_TOKENS),
        ideation_model=cfg.IDEATION_MODEL,
        ideation_max_tokens=int(cfg.IDEATION_MAX_TOKENS),
        rollout_evaluator_model=cfg.ROLLOUT_EVALUATOR_MODEL,
        web_search=bool(getattr(cfg, "WEB_SEARCH", False)),
        variation_dimensions=(
            tuple(variation_dimensions_raw) if variation_dimensions_raw else None
        ),
        selected_variations=(
            tuple(selected_variations_raw) if selected_variations_raw else None
        ),
        quality_keys=tuple(getattr(cfg, "QUALITY_KEYS", DEFAULT_QUALITY_KEYS)),
        seed=int(getattr(cfg, "SEED", 0)),
        max_lora_rank=int(getattr(cfg, "MAX_LORA_RANK", 64)),
        hf_repo=getattr(cfg, "HF_REPO", HF_REPO_ID),
    )


# ---------------------------------------------------------------------------
# Cell enumeration
# ---------------------------------------------------------------------------


def _enumerate_cells(nc: NormalisedConfig) -> list[CanonicalCell]:
    """Cartesian product of scales; dedupe after zero-dropping."""
    if not nc.adapters:
        return [CanonicalCell(entries=())]
    scale_lists = [nc.scales_per_adapter[a.slug] for a in nc.adapters]
    seen: set[tuple[tuple[str, float], ...]] = set()
    cells: list[CanonicalCell] = []
    for combo in itertools.product(*scale_lists):
        pairs = [(nc.adapters[i], float(combo[i])) for i in range(len(nc.adapters))]
        cell = CanonicalCell.from_scales(pairs)
        key = tuple((s.slug, sc) for s, sc in cell.entries)
        if key in seen:
            continue
        seen.add(key)
        cells.append(cell)
    return cells


# ---------------------------------------------------------------------------
# Fingerprint computation
# ---------------------------------------------------------------------------


def _ideation_fp_for_trait(
    nc: NormalisedConfig,
    trait: str,
    prompts: dict[str, str],
    behavior_description: str,
) -> str:
    return ideation_fingerprint(
        behavior_name=trait,
        behavior_description=behavior_description,
        understanding_model=nc.understanding_model,
        understanding_max_tokens=nc.understanding_max_tokens,
        ideation_model=nc.ideation_model,
        ideation_max_tokens=nc.ideation_max_tokens,
        num_scenarios=nc.num_scenarios,
        variation_dimensions=list(nc.variation_dimensions or []),
        web_search=nc.web_search,
        temperature=nc.ideation_temperature,
        evaluator_reasoning_effort=nc.ideation_reasoning_effort,
        understanding_prompts=prompts_subset(prompts, "understanding"),
        ideation_prompts=prompts_subset(prompts, "ideation"),
        seed=nc.seed,
    )


def _rollout_fp(
    nc: NormalisedConfig,
    ideation_fp: str,
    prompts: dict[str, str],
) -> str:
    return rollout_cell_fingerprint(
        ideation_fp=ideation_fp,
        scenario_version=nc.scenario_version,
        evaluator_model=nc.rollout_evaluator_model,
        modality=nc.modality,
        max_turns=nc.max_turns,
        rollout_max_tokens=nc.rollout_max_tokens,
        num_reps=nc.num_reps,
        no_user_mode=nc.no_user_mode,
        selected_variations=list(nc.selected_variations or []),
        anonymous_target=nc.anonymous_target,
        temperature=nc.temperature,
        evaluator_reasoning_effort=nc.evaluator_reasoning_effort,
        target_reasoning_effort=nc.target_reasoning_effort,
        rollout_prompts=prompts_subset(prompts, "rollout"),
        seed=nc.seed,
    )


# ---------------------------------------------------------------------------
# Per-trait ideation resolution
# ---------------------------------------------------------------------------


@dataclass
class TraitResolved:
    trait: str
    behavior_description: str
    judgment_prompt: str
    ideation_fp: str
    ideation_dir: Path
    ideation_hf_path: str


def _resolve_ideations(
    nc: NormalisedConfig,
    prompts: dict[str, str],
    *,
    requested_stages: list[str],
    no_upload: bool,
    dry_run: bool,
) -> dict[str, TraitResolved]:
    """Hydrate or materialise the ideation cache entry for each trait.

    For each trait:
      1. Build trait context (behavior description + judgment prompt).
      2. Compute ideation_fp.
      3. Try to hydrate from HF.
      4. If incomplete and ``understanding``/``ideation`` are both requested,
         run bloom understanding + ideation into a transient data dir, copy
         the outputs into the ideation cache, then upload.
      5. Fail fast if incomplete and those stages were not requested.
    """
    out: dict[str, TraitResolved] = {}
    for trait_key in nc.traits:
        trait_name, behavior_desc, judgment_prompt = build_trait_context(trait_key)
        ideation_fp = _ideation_fp_for_trait(nc, trait_name, prompts, behavior_desc)
        hf_path = ideation_hf_dir(
            eval_name=nc.eval_name,
            trait=trait_name,
            ideation_version=nc.scenario_version,
            ideation_fp=ideation_fp,
        )
        local_dir, complete = hydrate_ideation(
            scratch_root=SCRATCH_ROOT,
            eval_name=nc.eval_name,
            trait=trait_name,
            ideation_version=nc.scenario_version,
            ideation_fp=ideation_fp,
            repo_id=nc.hf_repo,
            skip_download=no_upload,
        )
        print(
            f"  [ideation] trait={trait_name} fp={ideation_fp} "
            f"{'hit' if complete else 'miss'} -> {local_dir}"
        )

        if not complete:
            need_run = (
                "understanding" in requested_stages
                and "ideation" in requested_stages
            )
            if dry_run:
                print(f"  [dry-run] would run bloom understanding+ideation for {trait_name}")
            elif not need_run:
                sys.exit(
                    f"Error: ideation cache miss for trait {trait_name!r} (fp={ideation_fp}) "
                    f"but --stages does not include both 'understanding' and 'ideation'. "
                    f"Cannot compute ideation."
                )
            else:
                _run_ideation_bloom(
                    nc,
                    trait=trait_name,
                    behavior_desc=behavior_desc,
                    judgment_prompt=judgment_prompt,
                    ideation_fp=ideation_fp,
                    ideation_local_dir=local_dir,
                )
                if not no_upload:
                    login_from_env()
                    upload_ideation(
                        local_dir=local_dir,
                        eval_name=nc.eval_name,
                        trait=trait_name,
                        ideation_version=nc.scenario_version,
                        ideation_fp=ideation_fp,
                        repo_id=nc.hf_repo,
                        commit_message=(
                            f"bloom ideation: {trait_name} v{nc.scenario_version} "
                            f"fp={ideation_fp}"
                        ),
                    )

        out[trait_key] = TraitResolved(
            trait=trait_name,
            behavior_description=behavior_desc,
            judgment_prompt=judgment_prompt,
            ideation_fp=ideation_fp,
            ideation_dir=local_dir,
            ideation_hf_path=hf_path,
        )
    return out


def _run_ideation_bloom(
    nc: NormalisedConfig,
    *,
    trait: str,
    behavior_desc: str,
    judgment_prompt: str,
    ideation_fp: str,
    ideation_local_dir: Path,
) -> None:
    """Invoke bloom's understanding + ideation stages for one trait.

    Outputs are copied into the ideation cache dir + ``ideation_meta.json``
    is written. Run in a transient ``bloom-data`` + ``bloom-results`` root
    under :data:`BLOOM_WORK_ROOT` so the repo-tracked ``bloom-data/`` is
    never mutated.
    """
    import shutil

    work_root = BLOOM_WORK_ROOT / f"ideation_{trait}_{ideation_fp}"
    work_root.mkdir(parents=True, exist_ok=True)
    bloom_results_dir = work_root / "bloom-results" / trait
    if bloom_results_dir.exists():
        shutil.rmtree(bloom_results_dir)

    seed_overrides: dict[str, Any] = {
        "behavior.name": trait,
        "understanding.model": nc.understanding_model,
        "understanding.max_tokens": nc.understanding_max_tokens,
        "ideation.model": nc.ideation_model,
        "ideation.max_tokens": nc.ideation_max_tokens,
        "ideation.num_scenarios": nc.num_scenarios,
        "temperature": nc.ideation_temperature,
        "evaluator_reasoning_effort": nc.ideation_reasoning_effort,
    }
    if nc.variation_dimensions is not None:
        seed_overrides["ideation.variation_dimensions"] = list(nc.variation_dimensions)
    behaviors_extra = {trait: behavior_desc}
    prompts_extra = {"judgment_system_additional": judgment_prompt}

    with patched_bloom_data(
        nc.bloom_data_dir,
        seed_overrides,
        behaviors_extra=behaviors_extra,
        prompts_extra=prompts_extra,
    ) as data_dir:
        # Bloom writes to data_dir.parent/bloom-results/{trait}. We want that
        # to be under work_root so we don't litter the repo — patched_bloom_data
        # gives us a tempdir so this is already safe.
        for stage in ("understanding", "ideation"):
            run_bloom_stage(data_dir, stage)

        # Bloom's working dir is data_dir.parent/bloom-results/{trait}.
        produced = data_dir.parent / "bloom-results" / trait
        for name in (UNDERSTANDING_FILENAME, IDEATION_FILENAME):
            src = produced / name
            if not src.exists():
                sys.exit(
                    f"Error: bloom did not produce {name} for trait {trait} "
                    f"under {produced}"
                )
            shutil.copy2(src, ideation_local_dir / name)

    meta = {
        "trait": trait,
        "scenario_version": nc.scenario_version,
        "ideation_fp": ideation_fp,
        "behavior_description": behavior_desc,
        "understanding_model": nc.understanding_model,
        "ideation_model": nc.ideation_model,
        "num_scenarios": nc.num_scenarios,
        "variation_dimensions": list(nc.variation_dimensions or []),
        "seed": nc.seed,
    }
    (ideation_local_dir / IDEATION_META_FILENAME).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n"
    )


# ---------------------------------------------------------------------------
# Per-cell rollout + judgment
# ---------------------------------------------------------------------------


def _run_rollouts_for_cells(
    nc: NormalisedConfig,
    *,
    trait: str,
    trait_resolved: TraitResolved,
    cells_to_rollout: list[CanonicalCell],
    cell_dirs: dict[CanonicalCell, Path],
    prompts: dict[str, str],
    no_vllm: bool,
    sweep_id: str,
) -> Path | None:
    """Bake targets, ensure vLLM, run bloom rollout for each cell sequentially.

    Returns the combo-bake root (for post-run cleanup) or ``None`` if no
    combo cells were baked.
    """
    combo_bake_root = BAKED_ROOT / sweep_id / trait
    single_bake_root = BAKED_ROOT / "singles" / trait

    # Separate single vs combo targets so single bakes persist across sweeps.
    # We still pass everything through bake_cells_for_bloom for uniform
    # models_entry construction, but we route single bakes to single_bake_root.
    baked_roots_used: set[Path] = set()
    baked = bake_cells_for_bloom(
        cells_to_rollout,
        base_model=nc.base_model,
        baked_root=combo_bake_root,
        max_lora_rank=nc.max_lora_rank,
    )
    for cell, target in baked.items():
        if target.baked_dir is not None:
            baked_roots_used.add(target.baked_dir.parent)

    models_extra = {
        target.short_name: target.models_entry for target in baked.values()
    }
    target_short_names = [baked[c].short_name for c in cells_to_rollout]

    ensure_vllm_running(target_short_names, models_extra, no_vllm, project_root)

    bloom_work = BLOOM_WORK_ROOT / f"rollout_{trait}_{sweep_id}"
    bloom_work.mkdir(parents=True, exist_ok=True)
    bloom_results_dir = bloom_work / "bloom-results" / trait
    bloom_results_dir.mkdir(parents=True, exist_ok=True)

    seed_ideation_into_bloom_results(trait_resolved.ideation_dir, bloom_results_dir)

    for cell in cells_to_rollout:
        target = baked[cell]
        clean_bloom_results_dir(bloom_results_dir)
        seed_ideation_into_bloom_results(trait_resolved.ideation_dir, bloom_results_dir)

        seed_overrides: dict[str, Any] = {
            "behavior.name": trait,
            "rollout.target": target.short_name,
            "rollout.model": nc.rollout_evaluator_model,
            "rollout.modality": nc.modality,
            "rollout.max_turns": nc.max_turns,
            "rollout.max_tokens": nc.rollout_max_tokens,
            "rollout.num_reps": nc.num_reps,
            "rollout.no_user_mode": nc.no_user_mode,
            "anonymous_target": nc.anonymous_target,
            "temperature": nc.temperature,
            "evaluator_reasoning_effort": nc.evaluator_reasoning_effort,
            "target_reasoning_effort": nc.target_reasoning_effort,
        }
        if nc.selected_variations is not None:
            seed_overrides["rollout.selected_variations"] = list(nc.selected_variations)
        behaviors_extra = {trait: trait_resolved.behavior_description}
        prompts_extra = {"judgment_system_additional": trait_resolved.judgment_prompt}

        print(f"\n  [rollout] {cell.variant_label()} (target={target.short_name})")

        with patched_bloom_data(
            nc.bloom_data_dir,
            seed_overrides,
            behaviors_extra=behaviors_extra,
            prompts_extra=prompts_extra,
            models_extra=models_extra,
        ) as data_dir:
            produced = data_dir.parent / "bloom-results" / trait
            produced.mkdir(parents=True, exist_ok=True)
            seed_ideation_into_bloom_results(trait_resolved.ideation_dir, produced)
            run_bloom_stage(data_dir, "rollout")
            route_bloom_outputs(produced, stage="rollout", cell_dir=cell_dirs[cell])

    return combo_bake_root if combo_bake_root.exists() else None


def _run_judgment_for_cell(
    nc: NormalisedConfig,
    *,
    trait: str,
    trait_resolved: TraitResolved,
    cell: CanonicalCell,
    cell_dir: Path,
    target_short_name: str,
    models_extra: dict[str, Any],
    missing_pairs: list[tuple[str, str]],
    rollout_cell_fp: str,
) -> set[str]:
    """Run bloom judgment for each missing judge and split outputs.

    Returns the set of judge models for which per-quality files were newly
    written. Each judge model is run at most once per cell (all qualities
    are extracted from a single judgment.json).
    """
    # Group missing pairs by judge — we only need one bloom judgment call
    # per judge, then split into all its qualities.
    missing_by_judge: dict[str, list[str]] = {}
    for judge, quality in missing_pairs:
        missing_by_judge.setdefault(judge, []).append(quality)

    touched_judges: set[str] = set()
    import shutil

    rollout_src = cell_dir / "rollouts" / "rollout.json"
    if not rollout_src.exists():
        print(f"  [judge] {cell.variant_label()}: no cached rollout.json, skipping all judges")
        return touched_judges

    for judge, qualities in missing_by_judge.items():
        seed_overrides: dict[str, Any] = {
            "behavior.name": trait,
            "rollout.target": target_short_name,
            "judgment.model": judge,
            "temperature": nc.judge_temperature,
            "evaluator_reasoning_effort": nc.evaluator_reasoning_effort,
        }
        behaviors_extra = {trait: trait_resolved.behavior_description}
        prompts_extra = {"judgment_system_additional": trait_resolved.judgment_prompt}

        print(f"  [judge] {cell.variant_label()} / {judge} -> qualities={qualities}")

        with patched_bloom_data(
            nc.bloom_data_dir,
            seed_overrides,
            behaviors_extra=behaviors_extra,
            prompts_extra=prompts_extra,
            models_extra=models_extra,
        ) as data_dir:
            # Re-seed into the patched tempdir's bloom-results location.
            patched_produced = data_dir.parent / "bloom-results" / trait
            patched_produced.mkdir(parents=True, exist_ok=True)
            seed_ideation_into_bloom_results(trait_resolved.ideation_dir, patched_produced)
            shutil.copy2(rollout_src, patched_produced / "rollout.json")
            for tr in (cell_dir / "rollouts").glob("transcript_*.json"):
                shutil.copy2(tr, patched_produced / tr.name)
            run_bloom_stage(data_dir, "judgment")
            judgment_json = patched_produced / "judgment.json"
            if not judgment_json.exists():
                print(f"  [judge] {cell.variant_label()}/{judge}: bloom did not emit judgment.json")
                continue
            split_judgment_into_qualities(
                judgment_json,
                out_dir=cell_dir,
                judge_model=judge,
                behavior_name=trait,
                ideation_fp=trait_resolved.ideation_fp,
                rollout_cell_fp=rollout_cell_fp,
                quality_keys=nc.quality_keys,
                judge_temperature=nc.judge_temperature,
            )
            touched_judges.add(judge)

    return touched_judges


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _write_cell_info(
    cell: CanonicalCell,
    *,
    cell_dir: Path,
    nc: NormalisedConfig,
    trait: str,
    ideation_fp: str,
    ideation_hf_path: str,
    rollout_cell_fp: str,
) -> None:
    info = {
        "eval_name": f"{nc.eval_name}_{trait}",
        "trait": trait,
        "tier": cell.tier,
        "variant_label": cell.variant_label(),
        "entries": [
            {"slug": spec.slug, "ref": spec.ref, "scale": scale}
            for spec, scale in cell.entries
        ],
        "scenario_version": nc.scenario_version,
        "ideation_fp": ideation_fp,
        "rollout_cell_fp": rollout_cell_fp,
        "base_model": nc.base_model,
    }
    (cell_dir / CELL_INFO_RELPATH).write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n"
    )

    ref = {
        "trait": trait,
        "version": nc.scenario_version,
        "ideation_fp": ideation_fp,
        "hf_path": ideation_hf_path,
    }
    (cell_dir / IDEATION_REF_RELPATH).write_text(
        json.dumps(ref, indent=2, ensure_ascii=False) + "\n"
    )


def main() -> None:
    flags = _parse_flags()
    cfg = _load_config_module(flags.config)

    diffs = check_sweep_defaults(cfg)
    if diffs:
        confirm_or_abort(diffs, allow_custom=flags.allow_custom_fingerprint)

    nc = _normalise_config(cfg, judgment_models_override=flags.judgment_models)
    requested_stages = list(flags.stages)
    cells = _enumerate_cells(nc)

    print(f"Bloom sweep: {len(cells)} cell(s) × {len(nc.traits)} trait(s) "
          f"× {len(nc.judgment_models)} judge(s)")
    print(f"  traits:  {list(nc.traits)}")
    print(f"  judges:  {list(nc.judgment_models)}")
    print(f"  stages:  {requested_stages}")
    print(f"  HF repo: {nc.hf_repo}")

    _config, _behaviors, prompts, models_config = load_bloom_config(nc.bloom_data_dir)

    # Upfront model validation against the evaluator allowlist, using the
    # real models.json so short names resolve to their LiteLLM IDs.
    pseudo_seed = {
        "understanding": {"model": nc.understanding_model},
        "ideation": {"model": nc.ideation_model},
        "rollout": {"model": nc.rollout_evaluator_model},
    }
    validate_evaluator_models(
        pseudo_seed, list(nc.judgment_models), models_config=models_config
    )

    traits_resolved = _resolve_ideations(
        nc,
        prompts,
        requested_stages=requested_stages,
        no_upload=flags.no_upload,
        dry_run=flags.dry_run,
    )

    rollout_fps: dict[str, str] = {
        trait_key: _rollout_fp(nc, traits_resolved[trait_key].ideation_fp, prompts)
        for trait_key in nc.traits
    }
    for trait_key in nc.traits:
        tr = traits_resolved[trait_key]
        print(
            f"  [cell-fp] trait={tr.trait} ideation_fp={tr.ideation_fp} "
            f"rollout_cell_fp={rollout_fps[trait_key]}"
        )

    if flags.dry_run:
        print("\n[dry-run] No stages will be executed.")
        return

    if not flags.no_upload:
        login_from_env()

    # -- Per-trait orchestration -------------------------------------------
    sweep_id = uuid.uuid4().hex[:8]
    touched_cells: list[
        tuple[CanonicalCell, Path, str, TraitResolved, str]
    ] = []  # (cell, cell_dir, trait_key, resolved, rollout_fp)
    combo_bake_roots_to_cleanup: set[Path] = set()

    required_judge_pairs = [
        (j, q) for j in nc.judgment_models for q in nc.quality_keys
    ]

    for trait_key in nc.traits:
        tr = traits_resolved[trait_key]
        rollout_fp = rollout_fps[trait_key]
        eval_name = f"{nc.eval_name}_{tr.trait}"

        # Hydrate every cell's artifacts for this trait.
        cell_dirs: dict[CanonicalCell, Path] = {}
        cell_status: dict[CanonicalCell, Any] = {}
        for cell in cells:
            local_dir, status = hydrate_cell(
                cell,
                scratch_root=SCRATCH_ROOT,
                model_slug=nc.base_model_slug,
                eval_name=eval_name,
                fingerprint=rollout_fp,
                repo_id=nc.hf_repo,
                required_judge_qualities=required_judge_pairs,
                skip_download=flags.no_upload,
            )
            cell_dirs[cell] = local_dir
            cell_status[cell] = status

        # -- Rollouts -------------------------------------------------------
        cells_needing_rollouts: list[CanonicalCell] = []
        if "rollout" in requested_stages and not flags.skip_rollouts:
            cells_needing_rollouts = [c for c in cells if not cell_status[c].has_rollouts]
            if cells_needing_rollouts:
                print(
                    f"\n== rollout [{tr.trait}] : "
                    f"{len(cells_needing_rollouts)}/{len(cells)} cell(s) need rollouts =="
                )
                combo_root = _run_rollouts_for_cells(
                    nc,
                    trait=tr.trait,
                    trait_resolved=tr,
                    cells_to_rollout=cells_needing_rollouts,
                    cell_dirs=cell_dirs,
                    prompts=prompts,
                    no_vllm=flags.no_vllm,
                    sweep_id=sweep_id,
                )
                if combo_root is not None:
                    combo_bake_roots_to_cleanup.add(combo_root)
                # Refresh status after rollouts
                for cell in cells_needing_rollouts:
                    _, cell_status[cell] = hydrate_cell(
                        cell,
                        scratch_root=SCRATCH_ROOT,
                        model_slug=nc.base_model_slug,
                        eval_name=eval_name,
                        fingerprint=rollout_fp,
                        repo_id=nc.hf_repo,
                        required_judge_qualities=required_judge_pairs,
                        skip_download=True,
                    )
            else:
                print(f"\n== rollout [{tr.trait}] : all {len(cells)} cell(s) cached ==")
        elif "rollout" not in requested_stages:
            # --stages judgment-only: require rollouts already present.
            missing = [c for c in cells if not cell_status[c].has_rollouts]
            if missing:
                sys.exit(
                    f"Error: --stages excludes 'rollout' but {len(missing)} cell(s) "
                    f"have no cached rollouts for trait {tr.trait}. Labels: "
                    f"{[c.variant_label() for c in missing[:5]]}"
                )

        # -- Judgment -------------------------------------------------------
        if "judgment" in requested_stages and not flags.skip_judge:
            # Rebuild per-cell bake metadata for judges that need to run.
            cells_for_judgment = [
                c for c in cells
                if cell_status[c].has_rollouts
                and cell_status[c].missing_judge_qualities(required_judge_pairs)
            ]
            if cells_for_judgment:
                print(
                    f"\n== judgment [{tr.trait}] : "
                    f"{len(cells_for_judgment)}/{len(cells)} cell(s) need judges =="
                )
                # Judgment doesn't need baked adapters (it's an evaluator-only
                # step) but bloom still wants a valid target model entry;
                # pass an empty models_extra — bloom reads target from the
                # cached rollout.json rather than re-running anything.
                baked = bake_cells_for_bloom(
                    cells_for_judgment,
                    base_model=nc.base_model,
                    baked_root=BAKED_ROOT / sweep_id / tr.trait,
                    max_lora_rank=nc.max_lora_rank,
                )
                models_extra = {t.short_name: t.models_entry for t in baked.values()}
                for cell in cells_for_judgment:
                    missing_pairs = cell_status[cell].missing_judge_qualities(
                        required_judge_pairs
                    )
                    _run_judgment_for_cell(
                        nc,
                        trait=tr.trait,
                        trait_resolved=tr,
                        cell=cell,
                        cell_dir=cell_dirs[cell],
                        target_short_name=baked[cell].short_name,
                        models_extra=models_extra,
                        missing_pairs=missing_pairs,
                        rollout_cell_fp=rollout_fp,
                    )

        # -- Write cell_info.json + ideation_ref.json + schedule for upload -
        for cell in cells:
            cd = cell_dirs[cell]
            if not (cd / "rollouts" / "rollout.json").exists():
                continue
            _write_cell_info(
                cell,
                cell_dir=cd,
                nc=nc,
                trait=tr.trait,
                ideation_fp=tr.ideation_fp,
                ideation_hf_path=tr.ideation_hf_path,
                rollout_cell_fp=rollout_fp,
            )
            touched_cells.append((cell, cd, trait_key, tr, rollout_fp))

    # -- Upload touched cells ---------------------------------------------
    if not flags.no_upload and touched_cells:
        login_from_env()
        for cell, cell_dir, trait_key, tr, rollout_fp in touched_cells:
            eval_name = f"{nc.eval_name}_{tr.trait}"
            upload_cell(
                cell,
                local_dir=cell_dir,
                model_slug=nc.base_model_slug,
                eval_name=eval_name,
                fingerprint=rollout_fp,
                repo_id=nc.hf_repo,
                commit_message=(
                    f"bloom cell: {tr.trait} fp={rollout_fp} "
                    f"{cell.variant_label()}"
                ),
                allow_patterns=list(UPLOAD_ALLOW_PATTERNS),
            )

    # -- Cleanup combo bakes ----------------------------------------------
    for root in combo_bake_roots_to_cleanup:
        try:
            cleanup_baked_dir(root)
        except Exception as exc:
            print(f"  [cleanup] failed to remove {root}: {exc}")

    print("\n-- COMPLETE --")


if __name__ == "__main__":
    main()
