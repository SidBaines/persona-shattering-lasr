#!/usr/bin/env python3
"""Cell-oriented TRAIT-benchmark sweep runner (single-adapter and combos).

Provides the same content-addressed cache framework for TRAIT-benchmark runs
that :mod:`scripts_dev.evals.llm_judge_sweep.runner_cells` provides for LLM
judge sweeps. Each *cell* (unique LoRA combination at specific scales) is
atomic, hydrated/uploaded as a unit, and lives at the canonical path given
by :class:`CanonicalCell.hf_dir` under ``eval_name="trait_logprobs"``.

Execution strategy: this runner does NOT reimplement the TRAIT executor. It
calls the existing :func:`run_eval_suite` once per adapter-group (baseline,
each single-adapter set, each combo set) and re-homes each produced per-
ModelSpec output into its canonical cell dir. Single-adapter groups use
:class:`ScaleSweep` so the base model + PEFT adapter load is amortised
across scale points.

Config shape (mirrors the judge sweep):

- **Combo configs** supply ``ADAPTERS: list[AdapterSpec]`` and
  ``SCALES_PER_ADAPTER: dict[slug, list[float]]``.
- **Legacy single-adapter configs** (``ADAPTER_REF`` + ``SCALE_POINTS``) are
  auto-promoted to the combo shape on load.

Plus TRAIT-benchmark fingerprint fields (see
:mod:`src_dev.evals.trait_sweep.defaults` for canonical values).
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
import math
import shutil
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from dotenv import load_dotenv

# Ensure project root is on sys.path.
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src_dev.evals.cell_sweep.cell_identity import (
    AdapterSpec,
    CanonicalCell,
    format_scale,
    sweep_hf_root,
)
from src_dev.evals.trait_sweep.defaults import (
    check_trait_defaults,
    confirm_or_abort,
)
from src_dev.evals.trait_sweep.fingerprint import trait_fingerprint
from src_dev.evals.trait_sweep.layout import (
    CELL_INFO_RELPATH,
    INSPECT_LOGS_RELDIR,
    RUN_INFO_RELPATH,
    cell_status_on_disk,
    hydrate_cell,
    upload_cell,
)
from src_dev.utils.hf_hub import (
    login_from_env,
    upload_folder_to_dataset_repo,
)

HF_REPO_ID = "persona-shattering-lasr/monorepo"
EVAL_NAME_DEFAULT = "trait_logprobs"
SCRATCH_ROOT = Path("scratch/monorepo")
STAGING_ROOT = Path("scratch/trait_sweep_staging")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_flags() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cell-oriented TRAIT-benchmark sweep runner."
    )
    p.add_argument(
        "--config", required=True,
        help="Python module path to the config constants.",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-upload", action="store_true")
    p.add_argument(
        "--allow-custom-fingerprint", action="store_true",
        help="Skip the canonical-defaults prompt for config drift.",
    )
    return p.parse_args()


def _load_config(module_path: str) -> ModuleType:
    return importlib.import_module(module_path)


# ---------------------------------------------------------------------------
# Config normalisation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NormalisedConfig:
    adapters: tuple[AdapterSpec, ...]
    scales_per_adapter: dict[str, tuple[float, ...]]  # keyed by AdapterSpec.slug
    eval_name: str
    base_model: str
    base_model_slug: str
    benchmark: str
    samples_per_trait: int
    trait_splits: tuple[str, ...]
    shuffle_choices: bool
    seed: int
    temperature: float
    prefill: str
    min_choice_mass: float
    dynamic_mass_filter: bool
    template: str | None
    max_tokens: int | None
    batch_size: int
    plot_title: str


def _normalise_config(cfg: ModuleType) -> NormalisedConfig:
    if hasattr(cfg, "ADAPTERS"):
        adapters = tuple(cfg.ADAPTERS)
        scales_per_adapter = {
            a.slug: tuple(float(s) for s in cfg.SCALES_PER_ADAPTER[a.slug])
            for a in adapters
        }
    elif hasattr(cfg, "ADAPTER_REF"):
        spec = AdapterSpec.from_ref(cfg.ADAPTER_REF)
        adapters = (spec,)
        scales_per_adapter = {
            spec.slug: tuple(float(s) for s in cfg.SCALE_POINTS),
        }
    else:
        # Baseline-only sweep: no adapters, no scales.
        adapters = ()
        scales_per_adapter = {}

    return NormalisedConfig(
        adapters=adapters,
        scales_per_adapter=scales_per_adapter,
        eval_name=getattr(cfg, "EVAL_NAME_CANONICAL", EVAL_NAME_DEFAULT),
        base_model=cfg.BASE_MODEL,
        base_model_slug=cfg.BASE_MODEL_SLUG,
        benchmark=cfg.BENCHMARK,
        samples_per_trait=int(cfg.SAMPLES_PER_TRAIT),
        trait_splits=tuple(cfg.TRAIT_SPLITS),
        shuffle_choices=bool(cfg.SHUFFLE_CHOICES),
        seed=int(cfg.SEED),
        temperature=float(cfg.TEMPERATURE),
        prefill=str(getattr(cfg, "PREFILL", "ANSWER: ")),
        min_choice_mass=float(cfg.MIN_CHOICE_MASS),
        dynamic_mass_filter=bool(cfg.DYNAMIC_MASS_FILTER),
        template=cfg.TEMPLATE,
        max_tokens=cfg.MAX_TOKENS,
        batch_size=int(getattr(cfg, "BATCH_SIZE", 128)),
        plot_title=getattr(cfg, "PLOT_TITLE", "TRAIT logprobs sweep"),
    )


def _fingerprint_fields(nc: NormalisedConfig) -> dict[str, Any]:
    return dict(
        base_model=nc.base_model,
        benchmark=nc.benchmark,
        samples_per_trait=nc.samples_per_trait,
        shuffle_choices=nc.shuffle_choices,
        seed=nc.seed,
        temperature=nc.temperature,
        prefill=nc.prefill,
        min_choice_mass=nc.min_choice_mass,
        dynamic_mass_filter=nc.dynamic_mass_filter,
        template=nc.template,
        max_tokens=nc.max_tokens,
    )


def _benchmark_args_for_trait(nc: NormalisedConfig, trait: str) -> dict[str, Any]:
    """Kwargs for ``InspectBenchmarkSpec.benchmark_args`` restricted to one trait.

    Each trait split is dispatched as its own eval spec so per-trait outputs
    land in distinct subdirs — enabling per-trait caching.
    """
    common: dict[str, Any] = {
        "samples_per_trait": nc.samples_per_trait,
        "trait_splits": [trait],
        "shuffle_choices": nc.shuffle_choices,
    }
    if nc.benchmark == "personality_trait_logprobs":
        return {
            **common,
            "prefill": nc.prefill,
            "min_choice_mass": nc.min_choice_mass,
            "dynamic_mass_filter": nc.dynamic_mass_filter,
            "template": nc.template,
        }
    if nc.benchmark == "personality_trait_sampled":
        args = dict(common)
        if nc.max_tokens is not None:
            args["max_tokens"] = nc.max_tokens
        return args
    raise ValueError(f"Unsupported TRAIT benchmark: {nc.benchmark!r}")


def _trait_spec_name(nc: NormalisedConfig, trait: str) -> str:
    """Inspect-spec name for a single trait eval — drives the output subdir."""
    return f"{nc.eval_name}__{trait}"


# ---------------------------------------------------------------------------
# Cell enumeration
# ---------------------------------------------------------------------------


def _enumerate_cells(nc: NormalisedConfig) -> list[CanonicalCell]:
    """Cartesian product over scale lists; dedupe after zero-dropping.

    ``{A=1, B=0}`` and ``{A=1, C=0}`` collapse to the same canonical cell
    ``{A=1}`` because zero-scale entries are dropped — dedupe to avoid
    repeating work.
    """
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
# Cell grouping for per-group suite execution
# ---------------------------------------------------------------------------


def _group_cells(
    cells: list[CanonicalCell],
) -> tuple[
    list[CanonicalCell],
    dict[str, list[CanonicalCell]],
    dict[str, list[CanonicalCell]],
]:
    """Split cells into (baseline, single-by-adapter-slug, combo-by-combo-slug)."""
    baseline: list[CanonicalCell] = []
    by_single: dict[str, list[CanonicalCell]] = defaultdict(list)
    by_combo: dict[str, list[CanonicalCell]] = defaultdict(list)
    for cell in cells:
        if cell.tier == "baseline":
            baseline.append(cell)
        elif cell.tier == "single_adapter":
            by_single[cell.entries[0][0].slug].append(cell)
        else:
            by_combo[cell.combo_slug].append(cell)
    return baseline, dict(by_single), dict(by_combo)


# ---------------------------------------------------------------------------
# Suite execution per group, re-homing outputs into canonical cell dirs
# ---------------------------------------------------------------------------


def _build_eval_specs_for_traits(
    nc: NormalisedConfig, traits: list[str]
) -> list[Any]:
    """Build one ``InspectBenchmarkSpec`` per trait so per-trait outputs land
    in distinct subdirs under ``staging_root/<run_name>/<model>/<spec_name>/``.
    """
    from src_dev.evals import InspectBenchmarkSpec

    specs: list[Any] = []
    for trait in traits:
        specs.append(
            InspectBenchmarkSpec(
                name=_trait_spec_name(nc, trait),
                benchmark=nc.benchmark,
                benchmark_args=_benchmark_args_for_trait(nc, trait),
                n_runs=1,
            )
        )
    return specs


def _run_group_baseline(
    nc: NormalisedConfig,
    staging_root: Path,
    baseline_cell_dir: Path,
    traits: list[str],
) -> None:
    """Run a baseline-only suite call (one spec per missing trait) and re-home
    each trait's logs into ``baseline_cell_dir/native/inspect_logs/<Trait>/``.
    """
    if not traits:
        return
    from src_dev.evals import ModelSpec, SuiteConfig
    from src_dev.evals.suite import run_eval_suite

    run_name = "baseline"
    cfg = SuiteConfig(
        base_model=None,
        models=[ModelSpec(name="base", base_model=nc.base_model, scale=None)],
        evals=_build_eval_specs_for_traits(nc, traits),
        temperature=nc.temperature,
        batch_size=nc.batch_size,
        output_root=staging_root,
        run_name=run_name,
        skip_completed=False,
        auto_analyze=False,
    )
    run_eval_suite(cfg)

    for trait in traits:
        src = staging_root / run_name / "base" / _trait_spec_name(nc, trait)
        if not src.exists():
            raise RuntimeError(
                f"[baseline] expected suite output at {src}, not found"
            )
        _rehome_trait_output(src, baseline_cell_dir, trait)


def _run_group_single_adapter(
    nc: NormalisedConfig,
    adapter: AdapterSpec,
    missing_per_cell: dict[CanonicalCell, list[str]],
    staging_root: Path,
    cell_dirs: dict[CanonicalCell, Path],
    *,
    claim_baseline_cell: CanonicalCell | None = None,
    baseline_cell_dir: Path | None = None,
    baseline_missing_traits: list[str] | None = None,
) -> None:
    """Run a single-adapter scale sweep (one spec per missing trait) and
    re-home each (scale, trait) output into the matching cell's
    per-trait subdir.

    If ``claim_baseline_cell`` is supplied, the suite's ``base`` output is
    also re-homed into ``baseline_cell_dir`` — avoids running a separate
    baseline-only suite when the scale sweep already produces a base pass.
    """
    from src_dev.evals import ScaleSweep, SuiteConfig
    from src_dev.evals.suite import run_eval_suite

    scales = sorted({c.entries[0][1] for c in missing_per_cell})
    traits_needed: set[str] = set()
    for miss in missing_per_cell.values():
        traits_needed.update(miss)
    if baseline_missing_traits:
        traits_needed.update(baseline_missing_traits)
    traits_sorted = sorted(traits_needed)

    run_name = f"single__{adapter.slug}"
    cfg = SuiteConfig(
        base_model=nc.base_model,
        adapter=adapter.ref,
        sweep=ScaleSweep(points=scales),
        evals=_build_eval_specs_for_traits(nc, traits_sorted),
        temperature=nc.temperature,
        batch_size=nc.batch_size,
        output_root=staging_root,
        run_name=run_name,
        skip_completed=False,
        auto_analyze=False,
    )
    run_eval_suite(cfg)

    for cell, missing_traits in missing_per_cell.items():
        scale = cell.entries[0][1]
        scale_tag = f"{scale:+.2f}".replace(".", "p")
        spec_name = f"lora_{scale_tag}x"
        for trait in missing_traits:
            src = staging_root / run_name / spec_name / _trait_spec_name(nc, trait)
            if not src.exists():
                raise RuntimeError(
                    f"[single:{adapter.slug}] expected suite output at {src}, not found"
                )
            _rehome_trait_output(src, cell_dirs[cell], trait)

    if (
        claim_baseline_cell is not None
        and baseline_cell_dir is not None
        and baseline_missing_traits
    ):
        for trait in baseline_missing_traits:
            src_base = staging_root / run_name / "base" / _trait_spec_name(nc, trait)
            if src_base.exists():
                _rehome_trait_output(src_base, baseline_cell_dir, trait)


def _run_group_combo(
    nc: NormalisedConfig,
    combo_slug: str,
    missing_per_cell: dict[CanonicalCell, list[str]],
    staging_root: Path,
    cell_dirs: dict[CanonicalCell, Path],
) -> None:
    """Run an explicit-models suite call covering every (cell, missing-trait)
    pair for a combo group, and re-home each per-trait output into the
    matching cell's per-trait subdir.
    """
    from src_dev.evals import AdapterConfig, ModelSpec, SuiteConfig
    from src_dev.evals.suite import run_eval_suite

    traits_needed: set[str] = set()
    for miss in missing_per_cell.values():
        traits_needed.update(miss)
    traits_sorted = sorted(traits_needed)

    run_name = f"combo__{combo_slug}"
    models = [
        ModelSpec(
            name=cell.variant_label(),
            base_model=nc.base_model,
            adapters=[
                AdapterConfig(path=spec.ref, scale=scale)
                for spec, scale in cell.entries
            ],
        )
        for cell in missing_per_cell
    ]
    cfg = SuiteConfig(
        base_model=None,
        models=models,
        evals=_build_eval_specs_for_traits(nc, traits_sorted),
        temperature=nc.temperature,
        batch_size=nc.batch_size,
        output_root=staging_root,
        run_name=run_name,
        skip_completed=False,
        auto_analyze=False,
    )
    run_eval_suite(cfg)

    for cell, missing_traits in missing_per_cell.items():
        spec_name = cell.variant_label()
        for trait in missing_traits:
            src = staging_root / run_name / spec_name / _trait_spec_name(nc, trait)
            if not src.exists():
                raise RuntimeError(
                    f"[combo:{combo_slug}] expected suite output at {src}, not found"
                )
            _rehome_trait_output(src, cell_dirs[cell], trait)


def _rehome_trait_output(
    src_eval_dir: Path, cell_dir: Path, trait: str
) -> None:
    """Copy a suite's per-(model, eval-spec) output for a single trait into
    ``cell_dir/native/inspect_logs/<trait>/``. Also promotes the source's
    ``run_info.json`` to the cell root the first time it's seen so
    ``cell_status_on_disk`` can verify the run succeeded.
    """
    cell_dir.mkdir(parents=True, exist_ok=True)

    src_run_info = src_eval_dir / RUN_INFO_RELPATH
    dst_run_info = cell_dir / RUN_INFO_RELPATH
    if src_run_info.exists():
        shutil.copy2(src_run_info, dst_run_info)

    src_logs = src_eval_dir / INSPECT_LOGS_RELDIR
    dst_trait_dir = cell_dir / INSPECT_LOGS_RELDIR / trait
    dst_trait_dir.mkdir(parents=True, exist_ok=True)
    if src_logs.is_dir():
        for f in src_logs.glob("*.json"):
            shutil.copy2(f, dst_trait_dir / f.name)


# ---------------------------------------------------------------------------
# Aggregation — per-cell, per-trait scores from Inspect logs
# ---------------------------------------------------------------------------


def _cell_trait_scores(cell_dir: Path) -> dict[str, float]:
    """Extract per-trait mean scores from a cell's per-trait Inspect logs.

    Walks ``native/inspect_logs/<Trait>/`` subdirs, parses the latest log in
    each, and merges the extracted score dicts. Returns a ``{trait: score}``
    mapping; empty if no logs are present or none can be parsed.
    """
    try:
        from src_dev.evals.personality.analyze_results import _extract_scores
    except Exception:
        return {}
    logs_dir = cell_dir / INSPECT_LOGS_RELDIR
    if not logs_dir.is_dir():
        return {}
    merged: dict[str, float] = {}
    for trait_dir in sorted(logs_dir.iterdir()):
        if not trait_dir.is_dir():
            continue
        logs = sorted(trait_dir.glob("*.json"))
        if not logs:
            continue
        result = _extract_scores(logs[-1])
        if result is None:
            continue
        scores, _parse_rate = result
        for k, v in scores.items():
            if isinstance(v, (int, float)) and not math.isnan(float(v)):
                merged[k] = float(v)
    return merged


def _aggregate(
    nc: NormalisedConfig,
    cells: list[CanonicalCell],
    cell_dirs: dict[CanonicalCell, Path],
    sweep_root: Path,
) -> Path:
    """Walk every cell, extract per-trait scores, write grid_summary.jsonl."""
    rows: list[dict[str, Any]] = []
    for cell in cells:
        scores = _cell_trait_scores(cell_dirs[cell])
        row: dict[str, Any] = {
            "cell_tag": cell.variant_label(),
            "cell_entries": [
                {"slug": s.slug, "scale": sc} for s, sc in cell.entries
            ],
            "tier": cell.tier,
            "scores": scores,
        }
        rows.append(row)
    out_path = sweep_root / "analysis" / "grid_summary.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote summary: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_1d(
    nc: NormalisedConfig,
    cells: list[CanonicalCell],
    cell_dirs: dict[CanonicalCell, Path],
    sweep_root: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not available; skipping.")
        return

    adapter = nc.adapters[0]
    cell_by_scale: dict[float, CanonicalCell] = {}
    for cell in cells:
        if not cell.entries:
            cell_by_scale[0.0] = cell
        else:
            cell_by_scale[cell.entries[0][1]] = cell
    scales = sorted(cell_by_scale.keys())

    plots_dir = sweep_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # One line per trait on a single plot.
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    trait_series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for scale in scales:
        cell = cell_by_scale[scale]
        scores = _cell_trait_scores(cell_dirs[cell])
        for trait, value in scores.items():
            trait_series[trait].append((scale, value))

    for trait, pairs in sorted(trait_series.items()):
        pairs = sorted(pairs)
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        ax.plot(xs, ys, marker="o", linewidth=2, label=trait)

    ax.axvline(0.0, color="black", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_title(f"{nc.plot_title} — {adapter.slug}")
    ax.set_xlabel("LoRA scale")
    ax.set_ylabel("Trait score")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out = plots_dir / "trait_sweep_1d.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote plot: {out}")


def _plot_2d(
    nc: NormalisedConfig,
    cells: list[CanonicalCell],
    cell_dirs: dict[CanonicalCell, Path],
    sweep_root: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[plot] matplotlib/numpy not available; skipping 2D plots.")
        return

    a, b = nc.adapters[0], nc.adapters[1]
    xs = sorted(nc.scales_per_adapter[a.slug])
    ys = sorted(nc.scales_per_adapter[b.slug])

    def find_cell(sa: float, sb: float) -> CanonicalCell:
        return CanonicalCell.from_scales([(a, sa), (b, sb)])

    # Collect all traits present across cells.
    traits: set[str] = set()
    for cell in cells:
        traits.update(_cell_trait_scores(cell_dirs[cell]).keys())

    plots_dir = sweep_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for trait in sorted(traits):
        mat = np.full((len(ys), len(xs)), math.nan, dtype=float)
        for i, sa in enumerate(xs):
            for j, sb in enumerate(ys):
                cell = find_cell(sa, sb)
                if cell not in cell_dirs:
                    continue
                scores = _cell_trait_scores(cell_dirs[cell])
                if trait in scores:
                    mat[j, i] = scores[trait]

        fig, ax = plt.subplots(figsize=(6.0, 5.0))
        im = ax.imshow(
            mat,
            origin="lower",
            aspect="auto",
            extent=(xs[0] - 0.5, xs[-1] + 0.5, ys[0] - 0.5, ys[-1] + 0.5),
        )
        ax.set_xticks(xs)
        ax.set_yticks(ys)
        ax.set_xlabel(f"{a.slug} scale")
        ax.set_ylabel(f"{b.slug} scale")
        ax.set_title(f"{trait} — {nc.plot_title}")
        fig.colorbar(im, ax=ax)
        for i, sa in enumerate(xs):
            for j, sb in enumerate(ys):
                if not math.isnan(mat[j, i]):
                    ax.text(sa, sb, f"{mat[j, i]:.2f}",
                            ha="center", va="center", color="white", fontsize=8)
        fig.tight_layout()
        out = plots_dir / f"heatmap_{trait}.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote plot: {out}")


def _make_plots(
    nc: NormalisedConfig,
    cells: list[CanonicalCell],
    cell_dirs: dict[CanonicalCell, Path],
    sweep_root: Path,
) -> None:
    n = len(nc.adapters)
    if n == 1:
        _plot_1d(nc, cells, cell_dirs, sweep_root)
    elif n == 2:
        _plot_2d(nc, cells, cell_dirs, sweep_root)
    else:
        print(f"[plot] N={n} adapters — JSON grid_summary only, no plot.")


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


def _write_cell_info(cell: CanonicalCell, cell_dir: Path, fingerprint: str) -> None:
    payload = {
        "tier": cell.tier,
        "variant_label": cell.variant_label(),
        "entries": [
            {
                "ref": s.ref,
                "slug": s.slug,
                "category": s.category,
                "trait": s.trait,
                "direction": s.direction,
                "version": s.version,
                "scale": sc,
            }
            for s, sc in cell.entries
        ],
        "fingerprint": fingerprint,
    }
    (cell_dir / CELL_INFO_RELPATH).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def _upload_cells(
    nc: NormalisedConfig,
    cells: list[CanonicalCell],
    cell_dirs: dict[CanonicalCell, Path],
    fingerprint: str,
) -> None:
    for cell in cells:
        _write_cell_info(cell, cell_dirs[cell], fingerprint)
        upload_cell(
            cell,
            local_dir=cell_dirs[cell],
            model_slug=nc.base_model_slug,
            eval_name=nc.eval_name,
            fingerprint=fingerprint,
            repo_id=HF_REPO_ID,
            commit_message=f"{nc.eval_name}: upload cell {cell.variant_label()}",
        )


def _upload_sweep_root(
    nc: NormalisedConfig,
    sweep_root: Path,
    fingerprint: str,
) -> None:
    hf_path = sweep_hf_root(
        list(nc.adapters),
        model_slug=nc.base_model_slug,
        eval_name=nc.eval_name,
        fingerprint=fingerprint,
    )
    upload_folder_to_dataset_repo(
        local_dir=sweep_root,
        repo_id=HF_REPO_ID,
        path_in_repo=hf_path,
        commit_message=f"{nc.eval_name}: upload sweep analysis + plots",
        allow_patterns=["plots/**", "analysis/**", "sweep_config.json"],
    )
    print(f"  [upload] sweep root → {HF_REPO_ID}/{hf_path}")


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def _print_dry_run(
    nc: NormalisedConfig, cells: list[CanonicalCell], fingerprint: str
) -> None:
    print("DRY RUN: Cell-oriented TRAIT sweep")
    print(f"  eval name        : {nc.eval_name}")
    print(f"  base model       : {nc.base_model} ({nc.base_model_slug})")
    print(f"  benchmark        : {nc.benchmark}")
    print(f"  samples/trait    : {nc.samples_per_trait}")
    print(f"  trait splits     : {list(nc.trait_splits)}")
    print(f"  shuffle choices  : {nc.shuffle_choices}")
    print(f"  seed             : {nc.seed}")
    print(f"  adapters         : {[a.slug for a in nc.adapters]}")
    for a in nc.adapters:
        print(f"    {a.slug}: scales={list(nc.scales_per_adapter[a.slug])}")
    print(f"  cells            : {len(cells)} canonical cells")
    for cell in cells:
        print(f"    [{cell.tier}] {cell.variant_label()}")
    print(f"  fingerprint      : {fingerprint}")
    sweep_hf = sweep_hf_root(
        list(nc.adapters), model_slug=nc.base_model_slug,
        eval_name=nc.eval_name, fingerprint=fingerprint,
    )
    print(f"  sweep HF root    : {sweep_hf}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    flags = _parse_flags()
    cfg = _load_config(flags.config)

    diffs = check_trait_defaults(cfg)
    confirm_or_abort(diffs, allow_custom=flags.allow_custom_fingerprint)

    nc = _normalise_config(cfg)
    load_dotenv()
    upload = not flags.no_upload
    if upload:
        login_from_env()

    fingerprint = trait_fingerprint(**_fingerprint_fields(nc))
    cells = _enumerate_cells(nc)

    if flags.dry_run:
        _print_dry_run(nc, cells, fingerprint)
        return

    # Stage 1 — hydrate every cell from HF.
    cell_dirs: dict[CanonicalCell, Path] = {}
    cell_status: dict[CanonicalCell, Any] = {}
    for cell in cells:
        local_dir, status = hydrate_cell(
            cell,
            scratch_root=SCRATCH_ROOT,
            model_slug=nc.base_model_slug,
            eval_name=nc.eval_name,
            fingerprint=fingerprint,
            repo_id=HF_REPO_ID,
            skip_download=not upload,
        )
        cell_dirs[cell] = local_dir
        cell_status[cell] = status

    required_traits = list(nc.trait_splits)

    def _missing_traits(cell: CanonicalCell) -> list[str]:
        present = cell_status[cell].present_traits
        return [t for t in required_traits if t not in present]

    n_fully = sum(1 for c in cells if not _missing_traits(c))
    print(
        f"[hydrate] {n_fully}/{len(cells)} cells fully cover "
        f"{len(required_traits)} required trait splits"
    )

    # Stage 2 — group-wise suite execution for (cell, trait) pairs that are missing.
    baseline_cells, by_single, by_combo = _group_cells(cells)
    staging_root = STAGING_ROOT / fingerprint
    staging_root.mkdir(parents=True, exist_ok=True)

    baseline_cell = baseline_cells[0] if baseline_cells else None
    baseline_missing_traits = _missing_traits(baseline_cell) if baseline_cell else []

    # Run single-adapter groups (claim baseline if still missing).
    claimed_baseline = False
    for adapter_slug, group_cells in sorted(by_single.items()):
        adapter = next(a for a in nc.adapters if a.slug == adapter_slug)
        missing_per_cell: dict[CanonicalCell, list[str]] = {}
        for c in group_cells:
            mt = _missing_traits(c)
            if mt:
                missing_per_cell[c] = mt
        claim_traits = None
        claim = None
        cell_dir_for_baseline = None
        if baseline_missing_traits and not claimed_baseline:
            claim = baseline_cell
            cell_dir_for_baseline = cell_dirs[baseline_cell]
            claim_traits = list(baseline_missing_traits)
            claimed_baseline = True
        if not missing_per_cell and not claim_traits:
            continue
        n_missing_traits = sum(len(v) for v in missing_per_cell.values())
        print(
            f"[run] single-adapter group {adapter_slug}: "
            f"{len(missing_per_cell)}/{len(group_cells)} cells with missing traits "
            f"({n_missing_traits} (cell,trait) pairs)"
            + (f" (+baseline: {len(claim_traits)} traits)" if claim_traits else "")
        )
        _run_group_single_adapter(
            nc,
            adapter,
            missing_per_cell,
            staging_root,
            cell_dirs,
            claim_baseline_cell=claim,
            baseline_cell_dir=cell_dir_for_baseline,
            baseline_missing_traits=claim_traits,
        )
        for c in missing_per_cell:
            cell_status[c] = cell_status_on_disk(cell_dirs[c])
        if claim is not None:
            cell_status[claim] = cell_status_on_disk(cell_dirs[claim])

    # Run baseline-only group if still missing.
    if baseline_missing_traits and not claimed_baseline:
        print(
            f"[run] baseline group: {len(baseline_missing_traits)} traits missing"
        )
        _run_group_baseline(
            nc,
            staging_root,
            cell_dirs[baseline_cell],
            baseline_missing_traits,
        )
        cell_status[baseline_cell] = cell_status_on_disk(cell_dirs[baseline_cell])

    # Run combo groups.
    for combo_slug, group_cells in sorted(by_combo.items()):
        missing_per_cell = {}
        for c in group_cells:
            mt = _missing_traits(c)
            if mt:
                missing_per_cell[c] = mt
        if not missing_per_cell:
            continue
        n_missing_traits = sum(len(v) for v in missing_per_cell.values())
        print(
            f"[run] combo group {combo_slug}: "
            f"{len(missing_per_cell)}/{len(group_cells)} cells with missing traits "
            f"({n_missing_traits} (cell,trait) pairs)"
        )
        _run_group_combo(nc, combo_slug, missing_per_cell, staging_root, cell_dirs)
        for c in missing_per_cell:
            cell_status[c] = cell_status_on_disk(cell_dirs[c])

    # Stage 3 — aggregate.
    sweep_root = SCRATCH_ROOT / sweep_hf_root(
        list(nc.adapters),
        model_slug=nc.base_model_slug,
        eval_name=nc.eval_name,
        fingerprint=fingerprint,
    )
    sweep_root.mkdir(parents=True, exist_ok=True)
    _aggregate(nc, cells, cell_dirs, sweep_root)

    # Stage 4 — plots.
    _make_plots(nc, cells, cell_dirs, sweep_root)

    # Stage 5 — upload.
    if upload:
        _upload_cells(nc, cells, cell_dirs, fingerprint)
        _upload_sweep_root(nc, sweep_root, fingerprint)

    print(f"Done. sweep_root={sweep_root}")


if __name__ == "__main__":
    main()
