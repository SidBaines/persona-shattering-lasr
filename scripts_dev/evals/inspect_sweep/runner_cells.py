#!/usr/bin/env python3
"""Cell-oriented sweep runner for general Inspect benchmarks (capabilities evals).

Provides the same content-addressed cache framework for arbitrary Inspect
benchmarks (MMLU, GPQA, TruthfulQA, …) that
:mod:`scripts_dev.evals.trait_sweep.runner_cells` provides for TRAIT. Each
*cell* (unique LoRA combination at specific scales) is atomic, hydrated/
uploaded as a unit, and lives at the canonical path given by
:class:`CanonicalCell.hf_dir` under the config-supplied ``EVAL_NAME``.

Execution strategy: the runner delegates per-cell work to the existing
:func:`run_eval_suite`, calling it once per adapter-group (baseline, each
single-adapter set, each combo set) and re-homing each produced
per-``(ModelSpec, InspectBenchmarkSpec)`` output into the matching cell's
per-benchmark subdir.

Config shape::

    BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
    BASE_MODEL_SLUG = "llama-3.1-8B-Instruct"
    ADAPTERS = [AdapterSpec.from_ref(...), ...]          # may be empty
    SCALES_PER_ADAPTER = {"<slug>": [0.5, 1.0, 1.5]}
    BENCHMARK_SPECS = [InspectBenchmarkSpec(...), ...]
    EVAL_NAME = "inspect_sweep"
    SEED = 42
    TEMPERATURE = 0.0
    BATCH_SIZE = 128
    PLOT_TITLE = "MMLU + GPQA sweep"

See ``configs/_template.py`` for a documented starting point.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from dotenv import load_dotenv

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src_dev.evals.cell_sweep.cell_identity import (
    AdapterSpec,
    CanonicalCell,
    sweep_hf_root,
)
from src_dev.evals.cell_sweep.runner import (
    enumerate_cells as _enumerate_cells_generic,
    load_config_module,
    parse_sweep_flags,
    upload_sweep_root as _upload_sweep_root_generic,
    write_cell_info,
)
from src_dev.evals.inspect_sweep.defaults import (
    check_inspect_defaults,
    confirm_or_abort,
)
from src_dev.evals.inspect_sweep.fingerprint import inspect_sweep_fingerprint
from src_dev.evals.trait_sweep.layout import (
    INSPECT_LOGS_RELDIR,
    RUN_INFO_RELPATH,
    cell_status_on_disk,
    hydrate_cell,
    upload_cell,
)
from src_dev.utils.hf_hub import login_from_env

HF_REPO_ID = "persona-shattering-lasr/monorepo"
EVAL_NAME_DEFAULT = "inspect_sweep"
SCRATCH_ROOT = Path("scratch/monorepo")
STAGING_ROOT = Path("scratch/inspect_sweep_staging")


# ---------------------------------------------------------------------------
# Config normalisation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NormalisedConfig:
    adapters: tuple[AdapterSpec, ...]
    scales_per_adapter: dict[str, tuple[float, ...]]
    eval_name: str
    base_model: str
    base_model_slug: str
    benchmark_specs: tuple[Any, ...]
    seed: int
    temperature: float
    batch_size: int
    plot_title: str


def _normalise_config(cfg: ModuleType) -> NormalisedConfig:
    adapters = tuple(getattr(cfg, "ADAPTERS", ()))
    scales_per_adapter = {
        a.slug: tuple(float(s) for s in cfg.SCALES_PER_ADAPTER[a.slug])
        for a in adapters
    }
    specs = tuple(cfg.BENCHMARK_SPECS)
    if not specs:
        raise ValueError("Config must define at least one BENCHMARK_SPECS entry")
    names = [s.name for s in specs]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate benchmark spec names: {names}")
    return NormalisedConfig(
        adapters=adapters,
        scales_per_adapter=scales_per_adapter,
        eval_name=getattr(cfg, "EVAL_NAME", EVAL_NAME_DEFAULT),
        base_model=cfg.BASE_MODEL,
        base_model_slug=cfg.BASE_MODEL_SLUG,
        benchmark_specs=specs,
        seed=int(cfg.SEED),
        temperature=float(cfg.TEMPERATURE),
        batch_size=int(getattr(cfg, "BATCH_SIZE", 128)),
        plot_title=getattr(cfg, "PLOT_TITLE", "Inspect benchmark sweep"),
    )


# ---------------------------------------------------------------------------
# Cell enumeration + grouping
# ---------------------------------------------------------------------------


def _enumerate_cells(nc: NormalisedConfig) -> list[CanonicalCell]:
    return _enumerate_cells_generic(nc.adapters, nc.scales_per_adapter)


def _group_cells(
    cells: list[CanonicalCell],
) -> tuple[
    list[CanonicalCell],
    dict[str, list[CanonicalCell]],
    dict[str, list[CanonicalCell]],
]:
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
# Suite execution per group
# ---------------------------------------------------------------------------


def _specs_for_missing(
    nc: NormalisedConfig, missing_names: list[str]
) -> list[Any]:
    """Return the subset of ``nc.benchmark_specs`` whose ``name`` is missing."""
    by_name = {s.name: s for s in nc.benchmark_specs}
    return [by_name[n] for n in missing_names if n in by_name]


def _run_group_baseline(
    nc: NormalisedConfig,
    staging_root: Path,
    baseline_cell_dir: Path,
    missing_specs: list[str],
) -> None:
    if not missing_specs:
        return
    from src_dev.evals import ModelSpec, SuiteConfig
    from src_dev.evals.suite import run_eval_suite

    run_name = "baseline"
    cfg = SuiteConfig(
        base_model=None,
        models=[ModelSpec(name="base", base_model=nc.base_model, scale=None)],
        evals=_specs_for_missing(nc, missing_specs),
        temperature=nc.temperature,
        batch_size=nc.batch_size,
        output_root=staging_root,
        run_name=run_name,
        skip_completed=False,
        auto_analyze=False,
    )
    run_eval_suite(cfg)

    for spec_name in missing_specs:
        src = staging_root / run_name / "base" / spec_name
        if not src.exists():
            raise RuntimeError(
                f"[baseline] expected suite output at {src}, not found"
            )
        _rehome_spec_output(src, baseline_cell_dir, spec_name)


def _run_group_single_adapter(
    nc: NormalisedConfig,
    adapter: AdapterSpec,
    missing_per_cell: dict[CanonicalCell, list[str]],
    staging_root: Path,
    cell_dirs: dict[CanonicalCell, Path],
    *,
    claim_baseline_cell: CanonicalCell | None = None,
    baseline_cell_dir: Path | None = None,
    baseline_missing_specs: list[str] | None = None,
) -> None:
    from src_dev.evals import ScaleSweep, SuiteConfig
    from src_dev.evals.suite import run_eval_suite

    scales = sorted({c.entries[0][1] for c in missing_per_cell})
    specs_needed: set[str] = set()
    for miss in missing_per_cell.values():
        specs_needed.update(miss)
    if baseline_missing_specs:
        specs_needed.update(baseline_missing_specs)
    specs_sorted = sorted(specs_needed)

    run_name = f"single__{adapter.slug}"
    cfg = SuiteConfig(
        base_model=nc.base_model,
        adapter=adapter.ref,
        sweep=ScaleSweep(points=scales),
        evals=_specs_for_missing(nc, specs_sorted),
        temperature=nc.temperature,
        batch_size=nc.batch_size,
        output_root=staging_root,
        run_name=run_name,
        skip_completed=False,
        auto_analyze=False,
    )
    run_eval_suite(cfg)

    for cell, missing_specs in missing_per_cell.items():
        scale = cell.entries[0][1]
        scale_tag = f"{scale:+.2f}".replace(".", "p")
        model_name = f"lora_{scale_tag}x"
        for spec_name in missing_specs:
            src = staging_root / run_name / model_name / spec_name
            if not src.exists():
                raise RuntimeError(
                    f"[single:{adapter.slug}] expected suite output at {src}, not found"
                )
            _rehome_spec_output(src, cell_dirs[cell], spec_name)

    if (
        claim_baseline_cell is not None
        and baseline_cell_dir is not None
        and baseline_missing_specs
    ):
        for spec_name in baseline_missing_specs:
            src_base = staging_root / run_name / "base" / spec_name
            if src_base.exists():
                _rehome_spec_output(src_base, baseline_cell_dir, spec_name)


def _run_group_combo(
    nc: NormalisedConfig,
    combo_slug: str,
    missing_per_cell: dict[CanonicalCell, list[str]],
    staging_root: Path,
    cell_dirs: dict[CanonicalCell, Path],
) -> None:
    from src_dev.evals import AdapterConfig, ModelSpec, SuiteConfig
    from src_dev.evals.suite import run_eval_suite

    specs_needed: set[str] = set()
    for miss in missing_per_cell.values():
        specs_needed.update(miss)
    specs_sorted = sorted(specs_needed)

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
        evals=_specs_for_missing(nc, specs_sorted),
        temperature=nc.temperature,
        batch_size=nc.batch_size,
        output_root=staging_root,
        run_name=run_name,
        skip_completed=False,
        auto_analyze=False,
    )
    run_eval_suite(cfg)

    for cell, missing_specs in missing_per_cell.items():
        model_name = cell.variant_label()
        for spec_name in missing_specs:
            src = staging_root / run_name / model_name / spec_name
            if not src.exists():
                raise RuntimeError(
                    f"[combo:{combo_slug}] expected suite output at {src}, not found"
                )
            _rehome_spec_output(src, cell_dirs[cell], spec_name)


def _rehome_spec_output(
    src_eval_dir: Path, cell_dir: Path, spec_name: str
) -> None:
    """Copy per-``(model, benchmark-spec)`` suite output into
    ``cell_dir/native/inspect_logs/<spec_name>/``. Also promotes the source's
    ``run_info.json`` to the cell root the first time it's seen so
    ``cell_status_on_disk`` can verify the run succeeded.
    """
    cell_dir.mkdir(parents=True, exist_ok=True)

    src_run_info = src_eval_dir / RUN_INFO_RELPATH
    dst_run_info = cell_dir / RUN_INFO_RELPATH
    if src_run_info.exists():
        shutil.copy2(src_run_info, dst_run_info)

    src_logs = src_eval_dir / INSPECT_LOGS_RELDIR
    dst_spec_dir = cell_dir / INSPECT_LOGS_RELDIR / spec_name
    dst_spec_dir.mkdir(parents=True, exist_ok=True)
    if src_logs.is_dir():
        for f in src_logs.glob("*.json"):
            shutil.copy2(f, dst_spec_dir / f.name)


# ---------------------------------------------------------------------------
# Aggregation + plots
# ---------------------------------------------------------------------------


def _cell_benchmark_scores(cell_dir: Path) -> dict[str, dict[str, float]]:
    """Walk a cell's per-benchmark Inspect logs and extract metric dicts.

    Returns ``{spec_name: {metric_name: value}}``. Empty if no parseable logs.
    """
    try:
        from src_dev.evals.personality.analyze_results import _extract_scores
    except Exception:
        return {}
    logs_dir = cell_dir / INSPECT_LOGS_RELDIR
    if not logs_dir.is_dir():
        return {}
    out: dict[str, dict[str, float]] = {}
    for spec_dir in sorted(logs_dir.iterdir()):
        if not spec_dir.is_dir():
            continue
        logs = sorted(spec_dir.glob("*.json"))
        if not logs:
            continue
        result = _extract_scores(logs[-1])
        if result is None:
            continue
        scores, _parse_rate = result
        keep = {
            k: float(v) for k, v in scores.items()
            if isinstance(v, (int, float)) and not math.isnan(float(v))
        }
        if keep:
            out[spec_dir.name] = keep
    return out


def _aggregate(
    nc: NormalisedConfig,
    cells: list[CanonicalCell],
    cell_dirs: dict[CanonicalCell, Path],
    sweep_root: Path,
) -> Path:
    rows: list[dict[str, Any]] = []
    for cell in cells:
        benchmark_scores = _cell_benchmark_scores(cell_dirs[cell])
        rows.append(
            {
                "cell_tag": cell.variant_label(),
                "cell_entries": [
                    {"slug": s.slug, "scale": sc} for s, sc in cell.entries
                ],
                "tier": cell.tier,
                "benchmark_scores": benchmark_scores,
            }
        )
    out_path = sweep_root / "analysis" / "grid_summary.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote summary: {out_path}")
    return out_path


def _primary_metric(metrics: dict[str, float]) -> tuple[str, float] | None:
    """Pick a single headline metric per benchmark for plotting.

    Prefers ``accuracy`` then ``choice``, else falls back to the first metric
    in the dict. Returns ``None`` when the dict is empty.
    """
    if not metrics:
        return None
    for key in ("accuracy", "choice", "exact_match"):
        if key in metrics:
            return key, metrics[key]
    first_key = next(iter(metrics))
    return first_key, metrics[first_key]


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

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for scale in scales:
        cell = cell_by_scale[scale]
        bench_scores = _cell_benchmark_scores(cell_dirs[cell])
        for spec_name, metrics in bench_scores.items():
            headline = _primary_metric(metrics)
            if headline is not None:
                series[spec_name].append((scale, headline[1]))

    for spec_name, pairs in sorted(series.items()):
        pairs = sorted(pairs)
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        ax.plot(xs, ys, marker="o", linewidth=2, label=spec_name)

    ax.axvline(0.0, color="black", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_title(f"{nc.plot_title} — {adapter.slug}")
    ax.set_xlabel("LoRA scale")
    ax.set_ylabel("Benchmark score")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out = plots_dir / "inspect_sweep_1d.png"
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
    else:
        print(f"[plot] N={n} adapters — JSON grid_summary only, no plot.")


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


def _upload_cells(
    nc: NormalisedConfig,
    cells: list[CanonicalCell],
    cell_dirs: dict[CanonicalCell, Path],
    fingerprint: str,
) -> None:
    for cell in cells:
        write_cell_info(cell, cell_dirs[cell], fingerprint)
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
    _upload_sweep_root_generic(
        sweep_root,
        hf_path=hf_path,
        repo_id=HF_REPO_ID,
        commit_message=f"{nc.eval_name}: upload sweep analysis + plots",
    )
    print(f"  [upload] sweep root → {HF_REPO_ID}/{hf_path}")


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def _print_dry_run(
    nc: NormalisedConfig, cells: list[CanonicalCell], fingerprint: str
) -> None:
    print("DRY RUN: Cell-oriented Inspect sweep")
    print(f"  eval name        : {nc.eval_name}")
    print(f"  base model       : {nc.base_model} ({nc.base_model_slug})")
    print(f"  benchmarks       : {[s.name for s in nc.benchmark_specs]}")
    print(f"  seed             : {nc.seed}")
    print(f"  temperature      : {nc.temperature}")
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
    flags = parse_sweep_flags("Cell-oriented Inspect-benchmark sweep runner.")
    cfg = load_config_module(flags.config)

    diffs = check_inspect_defaults(cfg)
    confirm_or_abort(diffs, allow_custom=flags.allow_custom_fingerprint)

    nc = _normalise_config(cfg)
    load_dotenv()
    upload = not flags.no_upload
    if upload:
        login_from_env()

    fingerprint = inspect_sweep_fingerprint(
        base_model=nc.base_model,
        benchmark_specs=nc.benchmark_specs,
        seed=nc.seed,
        temperature=nc.temperature,
    )
    cells = _enumerate_cells(nc)

    if flags.dry_run:
        _print_dry_run(nc, cells, fingerprint)
        return

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

    required_specs = [s.name for s in nc.benchmark_specs]

    def _missing_specs(cell: CanonicalCell) -> list[str]:
        present = cell_status[cell].present_traits
        return [n for n in required_specs if n not in present]

    n_fully = sum(1 for c in cells if not _missing_specs(c))
    print(
        f"[hydrate] {n_fully}/{len(cells)} cells fully cover "
        f"{len(required_specs)} required benchmarks"
    )

    baseline_cells, by_single, by_combo = _group_cells(cells)
    staging_root = STAGING_ROOT / fingerprint
    staging_root.mkdir(parents=True, exist_ok=True)

    baseline_cell = baseline_cells[0] if baseline_cells else None
    baseline_missing = _missing_specs(baseline_cell) if baseline_cell else []

    claimed_baseline = False
    for adapter_slug, group_cells in sorted(by_single.items()):
        adapter = next(a for a in nc.adapters if a.slug == adapter_slug)
        missing_per_cell: dict[CanonicalCell, list[str]] = {}
        for c in group_cells:
            miss = _missing_specs(c)
            if miss:
                missing_per_cell[c] = miss
        claim_names = None
        claim = None
        cell_dir_for_baseline = None
        if baseline_missing and not claimed_baseline:
            claim = baseline_cell
            cell_dir_for_baseline = cell_dirs[baseline_cell]
            claim_names = list(baseline_missing)
            claimed_baseline = True
        if not missing_per_cell and not claim_names:
            continue
        n_missing = sum(len(v) for v in missing_per_cell.values())
        print(
            f"[run] single-adapter group {adapter_slug}: "
            f"{len(missing_per_cell)}/{len(group_cells)} cells with missing benchmarks "
            f"({n_missing} (cell,benchmark) pairs)"
            + (f" (+baseline: {len(claim_names)} benchmarks)" if claim_names else "")
        )
        _run_group_single_adapter(
            nc,
            adapter,
            missing_per_cell,
            staging_root,
            cell_dirs,
            claim_baseline_cell=claim,
            baseline_cell_dir=cell_dir_for_baseline,
            baseline_missing_specs=claim_names,
        )
        for c in missing_per_cell:
            cell_status[c] = cell_status_on_disk(cell_dirs[c])
        if claim is not None:
            cell_status[claim] = cell_status_on_disk(cell_dirs[claim])

    if baseline_missing and not claimed_baseline:
        print(
            f"[run] baseline group: {len(baseline_missing)} benchmarks missing"
        )
        _run_group_baseline(
            nc,
            staging_root,
            cell_dirs[baseline_cell],
            baseline_missing,
        )
        cell_status[baseline_cell] = cell_status_on_disk(cell_dirs[baseline_cell])

    for combo_slug, group_cells in sorted(by_combo.items()):
        missing_per_cell = {}
        for c in group_cells:
            miss = _missing_specs(c)
            if miss:
                missing_per_cell[c] = miss
        if not missing_per_cell:
            continue
        n_missing = sum(len(v) for v in missing_per_cell.values())
        print(
            f"[run] combo group {combo_slug}: "
            f"{len(missing_per_cell)}/{len(group_cells)} cells with missing benchmarks "
            f"({n_missing} (cell,benchmark) pairs)"
        )
        _run_group_combo(nc, combo_slug, missing_per_cell, staging_root, cell_dirs)
        for c in missing_per_cell:
            cell_status[c] = cell_status_on_disk(cell_dirs[c])

    sweep_root = SCRATCH_ROOT / sweep_hf_root(
        list(nc.adapters),
        model_slug=nc.base_model_slug,
        eval_name=nc.eval_name,
        fingerprint=fingerprint,
    )
    sweep_root.mkdir(parents=True, exist_ok=True)
    _aggregate(nc, cells, cell_dirs, sweep_root)
    _make_plots(nc, cells, cell_dirs, sweep_root)

    if upload:
        _upload_cells(nc, cells, cell_dirs, fingerprint)
        _upload_sweep_root(nc, sweep_root, fingerprint)

    print(f"Done. sweep_root={sweep_root}")


if __name__ == "__main__":
    main()
