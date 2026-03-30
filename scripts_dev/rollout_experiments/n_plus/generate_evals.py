#!/usr/bin/env python3
"""N+ LoRA scale sweep — evaluation.

Scores pre-generated rollouts from generate_rollouts.py using the neuroticism
and coherence LLM judges.  Safe to re-run: already-scored evaluators are
skipped per cell automatically.  Add more evaluators to EVALUATIONS and
re-run to extend without re-scoring existing ones.

Usage::

    uv run python scripts_dev/rollout_experiments/n_plus/generate_evals.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# TODO: lora_scale_sweep was replaced by src_dev.sweep — these imports need
# updating before this script can be run against the current codebase.
from scripts_dev.rollout_experiments import evaluate_messages
from scripts_dev.rollout_experiments.lora_scale_sweep import (  # type: ignore[import]
    ScaleSweep,
    _scale_label,
)
from src_dev.persona_metrics.config import JudgeLLMConfig, PersonaMetricSpec
from src_dev.visualisations.plot_rollout_sweep import plot_sweep

# ── Config ─────────────────────────────────────────────────────────────────────

SWEEP_ID = "n_plus_lora_sweep"

# Set to the timestamped run directory produced by generate_rollouts.py, e.g.:
#   "20260318_143201_n_plus_lora_sweep"
# Leave as None to evaluate the most recently modified run.
RUN_NAME: str | None = None

SWEEP = ScaleSweep(min=-1.0, max=1.0, step=1.0)
CONDITIONS = ["no_prompt"]
PLOT_METRICS = [
    "overall/neuroticism.score/mean",
    "overall/coherence.score/mean",
]

EVALUATIONS: list[str | PersonaMetricSpec] = [
    PersonaMetricSpec(
        name="neuroticism",
        params={
            "judge_config": JudgeLLMConfig(
                provider="openrouter",
                model="openai/gpt-4o-mini",
                max_concurrent=32,
            )
        },
    ),
    PersonaMetricSpec(
        name="coherence",
        params={
            "judge_config": JudgeLLMConfig(
                provider="openrouter",
                model="openai/gpt-4o-mini",
                max_concurrent=32,
            )
        },
    ),
]


# ── Helpers ────────────────────────────────────────────────────────────────────


def _resolve_run_dir(sweep_root: Path, run_name: str | None) -> Path:
    if run_name is not None:
        return sweep_root / run_name
    candidates = sorted(
        (p for p in sweep_root.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {sweep_root}")
    return candidates[0]


def _resolve_experiment_dir(cell_dir: Path) -> Path:
    """Return the timestamped experiment subdir inside a sweep cell dir."""
    subdirs = [p for p in cell_dir.iterdir() if p.is_dir() and (p / "manifest.json").exists()]
    if not subdirs:
        raise FileNotFoundError(f"No experiment subdir with manifest.json found in {cell_dir}")
    return max(subdirs, key=lambda p: p.stat().st_mtime)


def _evals_already_done(cell_dir: Path, eval_names: list[str]) -> bool:
    """Return True only if all requested evals are present in run_info.json aggregates."""
    run_info_path = cell_dir / "run_info.json"
    if not run_info_path.exists():
        return False
    try:
        info = json.loads(run_info_path.read_text())
        aggs = info.get("aggregates") or {}
        return all(
            any(k.startswith(f"overall/{name}.") for k in aggs)
            for name in eval_names
        )
    except Exception:
        return False


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    load_dotenv()

    sweep_root = Path("scratch/runs") / SWEEP_ID
    output_root = _resolve_run_dir(sweep_root, RUN_NAME)
    print(f"Evaluating rollouts in: {output_root}\n")

    eval_names = [e if isinstance(e, str) else e.name for e in EVALUATIONS]
    scale_points = SWEEP.scale_points()

    for scale in scale_points:
        slabel = _scale_label(scale)
        for condition in CONDITIONS:
            cell_dir = output_root / slabel / condition
            if not cell_dir.exists():
                print(f"  skipping {slabel}/{condition} — directory not found")
                continue

            if _evals_already_done(cell_dir, eval_names):
                print(f"  skipping {slabel}/{condition} — already evaluated")
                continue

            print(f"  evaluating {slabel}/{condition} ...")
            try:
                experiment_dir = _resolve_experiment_dir(cell_dir)
                result = evaluate_messages(experiment_dir, EVALUATIONS)
                # Merge new aggregates into existing run_info.json.
                run_info_path = cell_dir / "run_info.json"
                if run_info_path.exists():
                    info = json.loads(run_info_path.read_text())
                    existing_aggs = info.get("aggregates") or {}
                    existing_aggs.update(result.aggregates)
                    info["aggregates"] = existing_aggs
                    run_info_path.write_text(json.dumps(info, indent=2))
                print(f"    -> {result.num_messages_evaluated} messages evaluated")
            except Exception as exc:  # noqa: BLE001
                print(f"    FAILED: {exc}")

    print("\nGenerating plots ...")
    for metric_key in PLOT_METRICS:
        metric_slug = metric_key.split("/")[1].split(".")[0]
        out_path = output_root / f"sweep_plot_{metric_slug}.png"
        try:
            plot_sweep(output_root, metric_key=metric_key, output=out_path)
            print(f"  plot saved to {out_path}")
        except Exception as exc:
            print(f"  plot failed for {metric_key}: {exc}")

    print(f"\nEvals complete. Results in {output_root}/")


if __name__ == "__main__":
    main()
