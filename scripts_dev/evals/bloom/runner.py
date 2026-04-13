"""Bloom eval orchestration wrapper with caching, multi-target/judge, and vLLM auto-launch.

See scripts_dev/evals/bloom/README.md for full documentation.

PIPELINE OVERVIEW
-----------------
bloom runs four stages in sequence:

  understanding -> ideation -> rollout -> judgment

Each stage gets a deterministic 12-hex run ID derived only from the config
fields that materially affect that stage's output, chained so each stage
implicitly depends on all upstream stages.  This means:

  - Changing the judgment model re-runs only judgment; rollout/ideation/
    understanding are reused from cache.
  - Changing the target model re-runs rollout + judgment only.
  - Changing ideation params re-runs ideation + rollout + judgment.
  - Use --seed N to get a fresh independent run of the same config.

Cache lookup order per stage:
  1. Local:  bloom-cache/evals/bloom/{stage}/{run_id}/
  2. Remote: HuggingFace dataset repo evals/bloom/{stage}/{run_id}/
  3. Run bloom -> save to local cache -> upload to HF

The original bloom-data/ directory is NEVER modified at runtime.  Each stage
runs against a temporary copy with overrides applied, so crashes leave no
residue in the config files.

COMMON INVOCATIONS
------------------
# Default: use models/trait from seed.yaml
uv run python scripts_dev/evals/bloom/runner.py

# Two targets, two judges
uv run python scripts_dev/evals/bloom/runner.py \\
    --targets llama-3.1-8b-it-base conscientiousness-low-llama \\
    --judgment-models gpt-5-mini gpt-5-nano

# Switch OCEAN trait (full name or single letter: c n o a e)
uv run python scripts_dev/evals/bloom/runner.py --trait neuroticism
uv run python scripts_dev/evals/bloom/runner.py --trait n

# New independent run of the same config
uv run python scripts_dev/evals/bloom/runner.py --seed 1

# Dry run: print run IDs without calling any APIs
uv run python scripts_dev/evals/bloom/runner.py --dry-run

# Re-judge existing rollouts with a new model (skip earlier stages)
uv run python scripts_dev/evals/bloom/runner.py \\
    --stages judgment --judgment-models kimi-k2

# Local cache only, no HF upload/download
uv run python scripts_dev/evals/bloom/runner.py --no-upload

# Disable vLLM auto-launch (fail fast with instructions if not running)
uv run python scripts_dev/evals/bloom/runner.py --no-vllm

CLI FLAGS
---------
--bloom-data PATH       Path to bloom-data directory (default: bloom-data)
--trait TRAIT           OCEAN trait to evaluate: conscientiousness/c, neuroticism/n,
                        openness/o, agreeableness/a, extraversion/e.
                        Auto-generates behavior description + judgment rubric from
                        persona_definitions.py.  Default: use seed.yaml behavior.name.
--targets MODEL ...     Target model short names from models.json.  Understanding and
                        ideation are shared; rollout/judgment run per target.
--judgment-models M ... Judge model short names.  Each gets its own run ID against
                        the same rollouts.
--stages STAGE ...      Subset of stages to run: understanding ideation rollout judgment.
                        Default: all four.
--seed INT              RNG seed for stochastic stages (default: 0).  Increment to
                        get a fresh independent run of the same config.
--hf-repo REPO          HuggingFace dataset repo for persistence
                        (default: persona-shattering-lasr/monorepo).
--no-upload             Disable HF upload/download; use local cache only.
--no-vllm               Disable automatic vLLM launch for local targets.
--dry-run               Print run IDs and exit without running anything.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv()

from src_dev.eval_stages import StageCache, StageCacheConfig, run_id_from_dict
from src_dev.evals.bloom.bloom_runtime import (
    ALLOWED_EVALUATOR_MODEL_IDS,
    PROMPT_KEYS_BY_STAGE as _PROMPT_KEYS_BY_STAGE,
    STAGES,
    build_trait_context as _build_trait_context,
    ensure_vllm_running,
    load_bloom_config,
    patched_bloom_data,
    prompts_subset as _prompts_subset,
    resolve_model_id as _resolve_model_id,
    run_bloom_stage,
    stage_data_dir as _stage_data_dir,
    validate_evaluator_models,
)
from src_dev.utils.hf_hub import login_from_env

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_REPO_DEFAULT = "persona-shattering-lasr/monorepo"
HF_BASE_PATH = "evals/bloom"


def compute_run_ids(
    config: dict[str, Any],
    behaviors: dict[str, str],
    prompts: dict[str, str],
    seed: int,
) -> dict[str, str]:
    """Compute deterministic, chained run IDs for each pipeline stage.

    Each stage ID depends only on:
      - The previous stage's run ID (establishing the dependency chain)
      - The config fields that directly affect that stage
      - The configurable-prompt keys relevant to that stage
      - The seed (for stochastic stages: ideation, rollout, judgment)

    Uses :func:`run_id_from_dict` from the shared ``eval_stages`` module
    for hash computation, which produces identical output to the previous
    ``_sha256_short`` implementation.
    """
    behavior_name = config["behavior"]["name"]
    temperature = config.get("temperature", 1.0)
    evaluator_effort = config.get("evaluator_reasoning_effort", "low")
    target_effort = config.get("target_reasoning_effort", "medium")

    understanding_id = run_id_from_dict({
        "stage": "understanding",
        "behavior": behavior_name,
        "behavior_description": behaviors[behavior_name],
        "model": config["understanding"]["model"],
        "max_tokens": config["understanding"]["max_tokens"],
        "temperature": temperature,
        "evaluator_reasoning_effort": evaluator_effort,
        "prompts": _prompts_subset(prompts, "understanding"),
    })

    ideation_id = run_id_from_dict({
        "stage": "ideation",
        "understanding_run_id": understanding_id,
        "model": config["ideation"]["model"],
        "num_scenarios": config["ideation"]["num_scenarios"],
        "variation_dimensions": sorted(config["ideation"].get("variation_dimensions") or []),
        "max_tokens": config["ideation"]["max_tokens"],
        "web_search": config["ideation"].get("web_search", False),
        "temperature": temperature,
        "evaluator_reasoning_effort": evaluator_effort,
        "prompts": _prompts_subset(prompts, "ideation"),
        "seed": seed,
    })

    rollout_id = run_id_from_dict({
        "stage": "rollout",
        "ideation_run_id": ideation_id,
        "evaluator_model": config["rollout"]["model"],
        "target_model": config["rollout"]["target"],
        "modality": config["rollout"]["modality"],
        "max_turns": config["rollout"]["max_turns"],
        "max_tokens": config["rollout"]["max_tokens"],
        "num_reps": config["rollout"]["num_reps"],
        "no_user_mode": config["rollout"].get("no_user_mode", False),
        "selected_variations": config["rollout"].get("selected_variations"),
        "anonymous_target": config.get("anonymous_target", False),
        "temperature": temperature,
        "evaluator_reasoning_effort": evaluator_effort,
        "target_reasoning_effort": target_effort,
        "prompts": _prompts_subset(prompts, "rollout"),
        "seed": seed,
    })

    judgment_id = run_id_from_dict({
        "stage": "judgment",
        "rollout_run_id": rollout_id,
        "model": config["judgment"]["model"],
        "additional_qualities": sorted(config["judgment"].get("additional_qualities") or []),
        "metajudgment_qualities": sorted(config["judgment"].get("metajudgment_qualities") or []),
        "max_tokens": config["judgment"]["max_tokens"],
        "num_samples": config["judgment"].get("num_samples", 1),
        "redaction_tags": config["judgment"].get("redaction_tags"),
        "anonymous_target": config.get("anonymous_target", False),
        "temperature": temperature,
        "evaluator_reasoning_effort": evaluator_effort,
        "prompts": _prompts_subset(prompts, "judgment"),
        "seed": seed,
    })

    return {
        "understanding": understanding_id,
        "ideation": ideation_id,
        "rollout": rollout_id,
        "judgment": judgment_id,
    }


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


def _restore_from_cache(cache_dir: Path, bloom_results_dir: Path) -> None:
    """Copy cached stage outputs into the bloom working directory."""
    bloom_results_dir.mkdir(parents=True, exist_ok=True)
    for src in cache_dir.iterdir():
        if src.name == "done.json":
            continue
        dst = bloom_results_dir / src.name
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def _save_to_cache(
    bloom_results_dir: Path, cache_dir: Path, patterns: list[str]
) -> None:
    """Copy specific bloom output files into the cache directory."""
    import fnmatch

    cache_dir.mkdir(parents=True, exist_ok=True)
    for item in bloom_results_dir.iterdir():
        if any(fnmatch.fnmatch(item.name, p) for p in patterns):
            dst = cache_dir / item.name
            if item.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(item, dst)
            else:
                shutil.copy2(item, dst)


def _run_one_stage(
    stage: str,
    run_id: str,
    bloom_data_dir: Path,
    bloom_results_dir: Path,
    cache: StageCache,
    behavior_name: str,
) -> None:
    """Check cache/HF for one stage run_id, running bloom only if needed.

    HF download/upload behavior is controlled by the ``StageCache`` config
    (``no_remote`` flag) — callers do not need to pass it separately.
    """
    marker = f"{stage}.json"
    stage_dir = cache.stage_dir(stage, run_id)
    print(f"-- {stage.upper()} (run_id={run_id}) --")

    # try_hydrate checks local cache first, then HF (respecting no_remote).
    if cache.try_hydrate(stage, run_id, marker=marker):
        print(f"  Cache hit -> restoring")
        _restore_from_cache(stage_dir, bloom_results_dir)
        return

    print(f"  Cache miss -> running stage")

    run_bloom_stage(bloom_data_dir, stage)

    # Determine which files to cache for this stage
    if stage == "rollout":
        patterns = ["transcript_*.json", "rollout.json"]
    else:
        patterns = [f"{stage}.json"]

    _save_to_cache(bloom_results_dir, stage_dir, patterns)

    # Write reproducibility marker (config provenance + git hash)
    cache.mark_complete(
        stage, run_id,
        config={"behavior": behavior_name, "stage": stage},
    )

    cache.upload(stage, run_id, commit_message=f"bloom eval - {behavior_name} - {stage} - {run_id}")

    print(f"  Done")


# ---------------------------------------------------------------------------
# LoRA scale sweep: adapter baking + dynamic model registration
# ---------------------------------------------------------------------------


def _sweep_target_name(scale: float) -> str:
    """Return the short model name for a sweep scale point."""
    return f"lora-scale-{scale:+.2f}"


def prepare_sweep_targets(
    adapter_ref: str,
    base_model: str,
    scale_points: list[float],
    baked_dir: Path,
    max_lora_rank: int = 64,
) -> tuple[list[str], dict[str, Any]]:
    """Bake LoRA adapters at each scale point and return dynamic model entries.

    Skips scale points whose baked adapter directory already exists on disk.
    Scale 0.0 is handled as the base model (no adapter, no baking needed).

    Returns:
        Tuple of (target_names, models_extra) where models_extra is a dict
        suitable for merging into models_config / models.json.
    """
    import gc

    import torch

    from src.utils.lora_baking import bake_lora_scale
    from src_dev.rollout_generation.model_providers import (
        _load_peft_model,
        _resolve_adapter_to_local,
    )

    baked_dir = Path(baked_dir)
    baked_dir.mkdir(parents=True, exist_ok=True)

    target_names: list[str] = []
    models_extra: dict[str, Any] = {}

    # Determine which non-zero scale points need baking
    scales_to_bake = [
        s for s in scale_points
        if s != 0.0 and not (baked_dir / f"scale_{s:+.2f}").exists()
    ]

    if scales_to_bake:
        print(f"  Baking {len(scales_to_bake)} adapter variant(s) to {baked_dir} ...")
        model, _tokenizer = _load_peft_model(
            base_model, adapter_ref, "default", "bfloat16"
        )
        for scale in scales_to_bake:
            out_dir = baked_dir / f"scale_{scale:+.2f}"
            print(f"    scale {scale:+.2f}: baking ...", flush=True)
            bake_lora_scale(model, "default", scale, out_dir)

        # Free PEFT model to avoid double GPU usage when vLLM launches
        try:
            model.cpu()
        except Exception:
            pass
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        n_existing = sum(1 for s in scale_points if s != 0.0)
        if n_existing > 0:
            print(f"  All {n_existing} adapter variant(s) already baked, skipping")

    # Build model entries for each scale point
    for scale in scale_points:
        name = _sweep_target_name(scale)
        target_names.append(name)

        if scale == 0.0:
            # Base model, no adapter
            models_extra[name] = {
                "id": f"openai/{name}",
                "org": "local",
                "name": f"Base model (scale 0.0)",
                "vllm": {"model": base_model},
            }
        else:
            adapter_path = str((baked_dir / f"scale_{scale:+.2f}").resolve())
            models_extra[name] = {
                "id": f"openai/{name}",
                "org": "local",
                "name": f"LoRA scale {scale:+.2f}",
                "vllm": {
                    "model": base_model,
                    "lora_path": adapter_path,
                    "max_lora_rank": max_lora_rank,
                },
            }

    return target_names, models_extra


def _config_for_target_and_judge(
    base_config: dict[str, Any], target: str | None, judge: str | None
) -> dict[str, Any]:
    """Return a deep-copied config with optional target and judge overrides."""
    cfg = copy.deepcopy(base_config)
    if target is not None:
        cfg["rollout"]["target"] = target
    if judge is not None:
        cfg["judgment"]["model"] = judge
    return cfg


def run_pipeline(
    bloom_data_dir: Path,
    seed: int,
    hf_repo: str,
    requested_stages: list[str],
    targets: list[str] | None,
    judgment_models: list[str] | None,
    dry_run: bool,
    no_upload: bool,
    no_vllm: bool = False,
    trait: str | None = None,
    # Model overrides (None = use seed.yaml default)
    understanding_model: str | None = None,
    ideation_model: str | None = None,
    rollout_evaluator_model: str | None = None,
    # LoRA scale sweep (all None = discrete target mode)
    adapter_ref: str | None = None,
    base_model: str | None = None,
    scale_points: list[float] | None = None,
    include_base: bool = True,
    baked_adapters_dir: Path | None = None,
    max_lora_rank: int = 64,
) -> None:
    """Run the full Bloom eval pipeline with caching and multi-target/judge support.

    Args:
        bloom_data_dir: Path to the bloom-data directory.
        seed: RNG seed for stochastic stages.
        hf_repo: HuggingFace dataset repo for persistence.
        requested_stages: Subset of stages to run.
        targets: Target model short names (or None to use seed.yaml default).
        judgment_models: Judge model short names (or None to use seed.yaml default).
        dry_run: If True, print run IDs and exit.
        no_upload: If True, disable HF upload/download.
        no_vllm: If True, disable automatic vLLM launch.
        trait: OCEAN trait override (or None to use seed.yaml default).
        understanding_model: Override understanding stage model.
        ideation_model: Override ideation stage model.
        rollout_evaluator_model: Override rollout evaluator model.
        adapter_ref: LoRA adapter reference for sweep mode (HF ``repo::subfolder``
            or local path).  When set, activates sweep mode.
        base_model: HuggingFace base model ID (required for sweep mode).
        scale_points: Scale factors for sweep mode.
        include_base: If True, add scale=0.0 to scale_points if missing.
        baked_adapters_dir: Directory for baked adapter cache.
        max_lora_rank: Max LoRA rank for vLLM config.
    """
    cache_config = StageCacheConfig(
        cache_root=bloom_data_dir.parent / "bloom-cache",
        hf_base_path=HF_BASE_PATH,
        hf_repo=hf_repo,
        no_remote=no_upload,
    )
    cache = StageCache(config=cache_config)

    config, behaviors, prompts, models_config = load_bloom_config(bloom_data_dir)

    # -- Trait override: mutate in-memory config/behaviors/prompts ----------
    # This ensures run IDs are computed from the trait-specific state, and the
    # same state is applied to every temp bloom-data copy used by each stage.
    g_seed_overrides: dict[str, Any] = {}
    g_behaviors_extra: dict[str, str] | None = None
    g_prompts_extra: dict[str, str] | None = None

    if trait:
        trait_name, behavior_desc, judgment_prompt = _build_trait_context(trait)
        config["behavior"]["name"] = trait_name
        behaviors[trait_name] = behavior_desc
        prompts["judgment_system_additional"] = judgment_prompt
        g_seed_overrides["behavior.name"] = trait_name
        g_behaviors_extra = {trait_name: behavior_desc}
        g_prompts_extra = {"judgment_system_additional": judgment_prompt}

    # -- Model overrides: mutate in-memory config + seed overrides -----------
    if understanding_model:
        config["understanding"]["model"] = understanding_model
        g_seed_overrides["understanding.model"] = understanding_model
    if ideation_model:
        config["ideation"]["model"] = ideation_model
        g_seed_overrides["ideation.model"] = ideation_model
    if rollout_evaluator_model:
        config["rollout"]["model"] = rollout_evaluator_model
        g_seed_overrides["rollout.model"] = rollout_evaluator_model

    behavior_name = config["behavior"]["name"]
    bloom_results_dir = bloom_data_dir.parent / "bloom-results" / behavior_name

    # -- LoRA scale sweep mode ------------------------------------------------
    sweep_mode = adapter_ref is not None
    g_models_extra: dict[str, Any] | None = None

    if sweep_mode:
        if base_model is None:
            sys.exit("Error: --base-model is required when --adapter-ref is set (sweep mode)")
        if not scale_points:
            sys.exit("Error: --scale-points is required when --adapter-ref is set (sweep mode)")

        # Ensure 0.0 is in the scale points when include_base is True
        if include_base and 0.0 not in scale_points:
            scale_points = sorted([0.0] + list(scale_points))

        if targets:
            print("Warning: --targets is ignored in sweep mode (targets are generated from scale points)")

        if not dry_run:
            sweep_targets, g_models_extra = prepare_sweep_targets(
                adapter_ref=adapter_ref,
                base_model=base_model,
                scale_points=scale_points,
                baked_dir=Path(baked_adapters_dir) if baked_adapters_dir else Path("scratch/bloom-baked-adapters"),
                max_lora_rank=max_lora_rank,
            )
        else:
            # For dry-run, generate target names without baking
            sweep_targets = [_sweep_target_name(s) for s in scale_points]
            g_models_extra = {
                name: {"id": f"openai/{name}", "org": "local", "name": f"LoRA scale {s:+.2f}", "vllm": {"model": base_model}}
                for s, name in zip(scale_points, sweep_targets)
            }

        # Merge dynamic entries into in-memory models_config
        models_config.update(g_models_extra)
        t_models = sweep_targets
    else:
        # Discrete target mode
        t_models = targets if targets else [config["rollout"]["target"]]

    # Resolve judgment models (fall back to whatever is in seed.yaml)
    j_models = judgment_models if judgment_models else [config["judgment"]["model"]]

    # Fail fast if any evaluator model is not in the allowlist
    validate_evaluator_models(config, j_models, models_config)

    # Pre-compute all run IDs for the summary
    base_ids = compute_run_ids(config, behaviors, prompts, seed)
    per_target_ids: dict[str, dict[str, Any]] = {}
    for target in t_models:
        per_target_ids[target] = {}
        for judge in j_models:
            cfg = _config_for_target_and_judge(config, target, judge)
            ids = compute_run_ids(cfg, behaviors, prompts, seed)
            per_target_ids[target][judge] = ids

    # -- Summary -----------------------------------------------------------
    print(f"Behavior : {behavior_name}")
    print(f"Seed     : {seed}")
    print(f"Targets  : {', '.join(t_models)}")
    print(f"Judges   : {', '.join(j_models)}")
    print()
    print(f"  {'understanding':<16} {base_ids['understanding']}")
    print(f"  {'ideation':<16} {base_ids['ideation']}")
    for target in t_models:
        rollout_id = per_target_ids[target][j_models[0]]["rollout"]
        print(f"  rollout [{target}]")
        print(f"    {'':2}{rollout_id}")
        for judge in j_models:
            jid = per_target_ids[target][judge]["judgment"]
            print(f"    judgment [{judge}]  {jid}")
    print()

    if dry_run:
        print("[dry-run] No stages will be executed.")
        return

    if not no_upload:
        login_from_env()

    # -- Understanding + Ideation (shared across all targets) ---------------
    for stage in ["understanding", "ideation"]:
        if stage not in requested_stages:
            continue
        with _stage_data_dir(bloom_data_dir, g_seed_overrides, g_behaviors_extra, g_prompts_extra, g_models_extra) as data_dir:
            _run_one_stage(
                stage, base_ids[stage],
                data_dir, bloom_results_dir, cache,
                behavior_name,
            )

    # -- vLLM health-check / auto-launch for local targets ------------------
    # Only launch vLLM if at least one rollout actually needs to be run
    # (i.e. not already satisfied by local cache or HF).
    if "rollout" in requested_stages and not dry_run:
        needs_rollout = any(
            not cache.is_complete("rollout", per_target_ids[t][j_models[0]]["rollout"], marker="rollout.json")
            for t in t_models
        )
        if needs_rollout:
            ensure_vllm_running(t_models, models_config, no_vllm, ROOT)
        else:
            print("  All rollouts cached -- skipping vLLM check")

    # -- Rollout + Judgment (once per target, judgment once per judge) -------
    for target in t_models:
        if len(t_models) > 1:
            print(f"\n== target: {target} ==")

        rollout_id = per_target_ids[target][j_models[0]]["rollout"]

        if "rollout" in requested_stages:
            with _stage_data_dir(bloom_data_dir, {**g_seed_overrides, "rollout.target": target}, g_behaviors_extra, g_prompts_extra, g_models_extra) as data_dir:
                _run_one_stage(
                    "rollout", rollout_id,
                    data_dir, bloom_results_dir, cache,
                    behavior_name,
                )

        if "judgment" in requested_stages:
            for judge in j_models:
                jid = per_target_ids[target][judge]["judgment"]
                if len(j_models) > 1:
                    print(f"\n  [ judge: {judge} ]")
                with _stage_data_dir(bloom_data_dir, {**g_seed_overrides, "rollout.target": target, "judgment.model": judge}, g_behaviors_extra, g_prompts_extra, g_models_extra) as data_dir:
                    _run_one_stage(
                        "judgment", jid,
                        data_dir, bloom_results_dir, cache,
                        behavior_name,
                    )

    print("\n-- COMPLETE --")

    plot_results(cache, per_target_ids, j_models, behavior_name)

    if sweep_mode and scale_points:
        plot_sweep_results(cache, per_target_ids, j_models, behavior_name, scale_points)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

# Additional quality keys to plot (subset of what bloom scores)
_QUALITY_KEYS = ["coherence"]
# OCEAN offset: bloom score -> OCEAN value
_OCEAN_OFFSET = 5


def _load_judgment_scores(
    cache: StageCache,
    per_target_ids: dict[str, dict[str, Any]],
    j_models: list[str],
    behavior_name: str,
) -> dict[tuple[str, str], dict[str, list[float]]]:
    """Load all scores from cached judgment.json files.

    Returns a dict keyed by (target, judge) -> {metric: [scores]}.
    Metrics: behavior_name (OCEAN scale) + _QUALITY_KEYS.
    """
    results: dict[tuple[str, str], dict[str, list[float]]] = {}
    for target, judge_ids in per_target_ids.items():
        for judge, ids in judge_ids.items():
            jid = ids["judgment"]
            path = cache.stage_dir("judgment", jid) / "judgment.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            scores: dict[str, list[float]] = {k: [] for k in [behavior_name] + _QUALITY_KEYS}
            for j in data.get("judgments", []):
                bp = j.get("behavior_presence")
                if bp is not None:
                    scores[behavior_name].append(float(bp) - _OCEAN_OFFSET)
                for q in _QUALITY_KEYS:
                    v = j.get(q)
                    if v is not None:
                        scores[q].append(float(v))
            results[(target, judge)] = scores
    return results


def plot_results(
    cache: StageCache,
    per_target_ids: dict[str, dict[str, Any]],
    j_models: list[str],
    behavior_name: str,
    output_root: Path | None = None,
) -> None:
    """Load judgment scores from cache and save three separate PNGs."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        import numpy as np
    except ImportError:
        print("  [plot] matplotlib not available -- skipping visualisation")
        return

    all_scores = _load_judgment_scores(cache, per_target_ids, j_models, behavior_name)
    if not all_scores:
        print("  [plot] No judgment results found in cache -- skipping visualisation")
        return

    t_models = list(dict.fromkeys(t for t, _ in all_scores))
    out_dir = output_root / behavior_name if output_root else Path("scratch/bloom-plots") / behavior_name
    out_dir.mkdir(parents=True, exist_ok=True)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    judge_color = {j: colors[i % len(colors)] for i, j in enumerate(j_models)}
    x = np.arange(len(t_models))
    n_judges = len(j_models)
    width = 0.7 / max(n_judges, 1)
    bins = np.arange(-4.5, 5.0, 1.0)

    def _short(name: str, n: int = 20) -> str:
        return name.split("/")[-1][:n]

    # -- Figure 1: mean OCEAN score ----------------------------------------
    fig1, ax_bar = plt.subplots(figsize=(max(7, 2.5 * len(t_models) + 2), 5))
    for ji, judge in enumerate(j_models):
        means, errs = [], []
        for target in t_models:
            sc = all_scores.get((target, judge), {}).get(behavior_name, [])
            means.append(float(np.mean(sc)) if sc else float("nan"))
            errs.append(float(np.std(sc)) if sc else 0.0)
        offset = (ji - (n_judges - 1) / 2) * width
        ax_bar.bar(x + offset, means, width * 0.9, yerr=errs, capsize=3,
                   label=judge, color=judge_color[judge], alpha=0.85)
        for ti, target in enumerate(t_models):
            sc = all_scores.get((target, judge), {}).get(behavior_name, [])
            if sc:
                ax_bar.scatter([x[ti] + offset] * len(sc), sc,
                               color=judge_color[judge], s=18, zorder=5, alpha=0.6)
    ax_bar.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(t_models, rotation=15, ha="right", fontsize=9)
    ax_bar.set_ylabel(f"OCEAN {behavior_name}\n(bloom score - 5)")
    ax_bar.set_title(f"{behavior_name} -- mean OCEAN score by target & judge")
    ax_bar.set_ylim(-4.5, 4.5)
    ax_bar.yaxis.set_major_locator(mticker.MultipleLocator(1))
    ax_bar.legend(fontsize=8, title="Judge", title_fontsize=8)
    fig1.tight_layout()
    p1 = out_dir / "scores_mean.png"
    fig1.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # -- Figure 2: histograms -- one row per target, judges as grouped bars -
    n_targets = len(t_models)
    bin_centers = (bins[:-1] + bins[1:]) / 2  # integer OCEAN values -4..+4
    bar_width = 0.8 / max(n_judges, 1)
    fig2, hist_axes = plt.subplots(
        n_targets, 1,
        figsize=(9, 3 * n_targets),
        sharex=True, squeeze=False,
    )
    for ti, target in enumerate(t_models):
        ax_h = hist_axes[ti][0]
        for ji, judge in enumerate(j_models):
            sc = all_scores.get((target, judge), {}).get(behavior_name, [])
            if sc:
                counts, _ = np.histogram(sc, bins=bins)
                offset = (ji - (n_judges - 1) / 2) * bar_width
                ax_h.bar(bin_centers + offset, counts, bar_width * 0.95,
                         label=judge, color=judge_color[judge], alpha=0.85, edgecolor="white")
        ax_h.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax_h.set_xlim(-4.5, 4.5)
        ax_h.xaxis.set_major_locator(mticker.MultipleLocator(1))
        ax_h.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax_h.set_ylabel("Count", fontsize=8)
        ax_h.set_title(_short(target), fontsize=9)
        ax_h.legend(fontsize=7, title="Judge", title_fontsize=7)
    hist_axes[-1][0].set_xlabel(f"OCEAN {behavior_name} score", fontsize=9)
    fig2.suptitle(f"{behavior_name} -- score distributions", fontsize=11, fontweight="bold")
    fig2.tight_layout()
    p2 = out_dir / "scores_hist.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # -- Figure 3: quality metrics -----------------------------------------
    if _QUALITY_KEYS:
        fig3, q_axes = plt.subplots(1, len(_QUALITY_KEYS),
                                    figsize=(4 * len(_QUALITY_KEYS), 4), squeeze=False)
        for qi, qkey in enumerate(_QUALITY_KEYS):
            ax_q = q_axes[0][qi]
            for ji, judge in enumerate(j_models):
                means = []
                for target in t_models:
                    sc = all_scores.get((target, judge), {}).get(qkey, [])
                    means.append(float(np.mean(sc)) if sc else float("nan"))
                offset = (ji - (n_judges - 1) / 2) * width
                ax_q.bar(x + offset, means, width * 0.9,
                         label=judge, color=judge_color[judge], alpha=0.85)
            ax_q.set_xticks(x)
            ax_q.set_xticklabels([_short(t, 15) for t in t_models],
                                  rotation=20, ha="right", fontsize=8)
            ax_q.set_title(qkey.replace("_", " "), fontsize=10)
            ax_q.set_ylim(0, 10)
            ax_q.yaxis.set_major_locator(mticker.MultipleLocator(2))
            ax_q.set_ylabel("Mean score (1-10)", fontsize=8)
            ax_q.legend(fontsize=7, title="Judge", title_fontsize=7)
        fig3.suptitle(f"{behavior_name} -- quality metrics", fontsize=11, fontweight="bold")
        fig3.tight_layout()
        p3 = out_dir / "scores_quality.png"
        fig3.savefig(p3, dpi=150, bbox_inches="tight")
        plt.close(fig3)
        print(f"\n  Plots saved -> {p1.name}, {p2.name}, {p3.name}  (in {out_dir})")
    else:
        print(f"\n  Plots saved -> {p1.name}, {p2.name}  (in {out_dir})")


def plot_sweep_results(
    cache: StageCache,
    per_target_ids: dict[str, dict[str, Any]],
    j_models: list[str],
    behavior_name: str,
    scale_points: list[float],
    output_root: Path | None = None,
) -> None:
    """Plot scale-vs-OCEAN-score curve for a LoRA scale sweep.

    Produces a line plot with error bars showing how the multi-turn
    behavioral score varies with adapter scale.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  [sweep-plot] matplotlib not available -- skipping")
        return

    all_scores = _load_judgment_scores(cache, per_target_ids, j_models, behavior_name)
    if not all_scores:
        print("  [sweep-plot] No judgment results found -- skipping")
        return

    from src_dev.evals.personality.analyze_results import _interval_ci_from_bootstrap

    out_dir = output_root or (ROOT / "bloom-results" / behavior_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ji, judge in enumerate(j_models):
        xs, ys, yerr_lo, yerr_hi = [], [], [], []
        for scale in scale_points:
            target_name = _sweep_target_name(scale)
            sc = all_scores.get((target_name, judge), {}).get(behavior_name, [])
            if sc:
                mean = float(np.mean(sc))
                ci_lo, ci_hi = _interval_ci_from_bootstrap(
                    np.array(sc), confidence=95, n_resamples=1000, seed=42
                )
                xs.append(scale)
                ys.append(mean)
                yerr_lo.append(max(0.0, mean - ci_lo))
                yerr_hi.append(max(0.0, ci_hi - mean))
            else:
                xs.append(scale)
                ys.append(float("nan"))
                yerr_lo.append(0.0)
                yerr_hi.append(0.0)

        color = colors[ji % len(colors)]
        ax.errorbar(
            xs, ys, yerr=[yerr_lo, yerr_hi],
            marker="o", linewidth=2, capsize=4, capthick=1.2,
            label=judge, color=color, alpha=0.9,
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(0, color="grey", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.set_xlabel("LoRA scale")
    ax.set_ylabel(f"OCEAN {behavior_name}\n(bloom score - 5)")
    ax.set_title(f"{behavior_name} -- bloom sweep (multi-turn)")
    ax.set_ylim(-4.5, 4.5)
    ax.legend(fontsize=8, title="Judge", title_fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    plot_path = out_dir / "scores_sweep.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Sweep plot saved -> {plot_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_config_module(module_path: str) -> Any:
    """Import and return a config module by dotted path."""
    import importlib
    return importlib.import_module(module_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config", default=None, metavar="MODULE",
        help="Python module path to a config file "
             "(e.g. scripts_dev.evals.bloom.configs.default). "
             "Constants from the module set defaults; CLI flags override them.",
    )
    parser.add_argument(
        "--bloom-data", default=None,
        help="Path to bloom-data directory (default: bloom-data)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="RNG seed included in run IDs for stochastic stages. "
             "Increment to get an independent run of the same config.",
    )
    parser.add_argument(
        "--hf-repo", default=None,
        help=f"HuggingFace dataset repo for persistence (default: {HF_REPO_DEFAULT})",
    )
    parser.add_argument(
        "--stages", nargs="+", default=STAGES, choices=STAGES,
        help="Stages to include (default: all). Skipped stages are never re-run "
             "even if their run ID changed.",
    )
    parser.add_argument(
        "--targets", nargs="+", default=None, metavar="MODEL",
        help="Run rollout+judgment for each target model (short name from models.json "
             "or direct LiteLLM ID). Understanding and ideation are shared and run "
             "only once. Default: use the target in seed.yaml.",
    )
    parser.add_argument(
        "--judgment-models", nargs="+", default=None, metavar="MODEL",
        help="Run judgment once per model (short name from models.json or direct "
             "LiteLLM ID). Each gets its own run ID; the shared rollout is reused "
             "from cache. Default: use the model in seed.yaml.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print run IDs and exit without running anything.",
    )
    parser.add_argument(
        "--no-upload", action="store_true",
        help="Disable HF upload/download; use local cache only.",
    )
    parser.add_argument(
        "--no-vllm", action="store_true",
        help="Disable automatic vLLM launch for local target models. "
             "The script will error with instructions if vLLM is not already running.",
    )
    parser.add_argument(
        "--trait", default=None, metavar="TRAIT",
        help="OCEAN trait to evaluate, e.g. 'conscientiousness', 'c', 'neuroticism', 'n', "
             "'openness', 'o', 'agreeableness', 'a', 'extraversion', 'e'. "
             "Overrides behavior.name in seed.yaml and auto-generates the behavior "
             "description and judgment rubric from persona_definitions.py. "
             "Default: use the behavior.name already set in seed.yaml.",
    )
    # -- Sweep mode flags ----------------------------------------------------
    parser.add_argument(
        "--adapter-ref", default=None, metavar="REF",
        help="LoRA adapter reference (HF repo::subfolder or local path). "
             "Activates sweep mode: bake the adapter at each scale point and "
             "run bloom rollout+judgment per scale.",
    )
    parser.add_argument(
        "--base-model", default=None, metavar="MODEL",
        help="HuggingFace base model ID for sweep mode "
             "(e.g. meta-llama/Llama-3.1-8B-Instruct).",
    )
    parser.add_argument(
        "--scale-points", nargs="+", type=float, default=None, metavar="SCALE",
        help="Scale points for sweep mode (e.g. --scale-points -2.0 -1.0 0.0 1.0 2.0).",
    )
    parser.add_argument(
        "--baked-adapters-dir", default=None, metavar="DIR",
        help="Directory for baked adapter cache (default: scratch/bloom-baked-adapters).",
    )
    args = parser.parse_args()

    # -- Resolve config module defaults, then override with CLI flags ---------
    cfg = _load_config_module(args.config) if args.config else None

    def _cfg_attr(name: str, default: Any = None) -> Any:
        """Return cfg.NAME if it exists, else default."""
        return getattr(cfg, name, default) if cfg else default

    bloom_data = args.bloom_data or str(_cfg_attr("BLOOM_DATA_DIR", "bloom-data"))
    seed = args.seed if args.seed is not None else (_cfg_attr("SEED") or 0)
    hf_repo = args.hf_repo or _cfg_attr("HF_REPO", HF_REPO_DEFAULT)
    trait = args.trait or _cfg_attr("TRAIT")
    targets = args.targets or _cfg_attr("TARGETS") or None
    judgment_models = args.judgment_models or _cfg_attr("JUDGMENT_MODELS") or None

    # Model overrides (only from config module, no CLI flags)
    understanding_model = _cfg_attr("UNDERSTANDING_MODEL")
    ideation_model = _cfg_attr("IDEATION_MODEL")
    rollout_evaluator_model = _cfg_attr("ROLLOUT_EVALUATOR_MODEL")

    # Sweep mode: CLI flags override config module
    adapter_ref = args.adapter_ref or _cfg_attr("ADAPTER_REF")
    base_model = args.base_model or _cfg_attr("BASE_MODEL")
    scale_points = args.scale_points or _cfg_attr("SCALE_POINTS")
    include_base = _cfg_attr("INCLUDE_BASE", True)
    baked_adapters_dir = args.baked_adapters_dir or _cfg_attr("BAKED_ADAPTERS_DIR")
    max_lora_rank = _cfg_attr("MAX_LORA_RANK", 64)

    run_pipeline(
        bloom_data_dir=Path(bloom_data).resolve(),
        seed=seed,
        hf_repo=hf_repo,
        requested_stages=args.stages,
        targets=targets,
        judgment_models=judgment_models,
        dry_run=args.dry_run,
        no_upload=args.no_upload,
        no_vllm=args.no_vllm,
        trait=trait,
        understanding_model=understanding_model,
        ideation_model=ideation_model,
        rollout_evaluator_model=rollout_evaluator_model,
        adapter_ref=adapter_ref,
        base_model=base_model,
        scale_points=scale_points,
        include_base=include_base,
        baked_adapters_dir=Path(baked_adapters_dir) if baked_adapters_dir else None,
        max_lora_rank=max_lora_rank,
    )


if __name__ == "__main__":
    main()
