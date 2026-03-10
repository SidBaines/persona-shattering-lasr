"""LoRA scale sweep over multi-phase rollout experiments.

Provides ``RolloutScaleSweep`` — mirrors the Inspect eval suite's sweep pattern
for the rollout generation stack:

  1. Load base model + LoRA adapter **once**.
  2. Outer loop: scale points.  Apply ``LoRaScaling`` in-place.
  3. Inner loop: conditions (phase lists).  Run ``run_experiment`` for each.
  4. Restore ``LoRaScaling``.  Move to next scale.
  5. Release model after all scale points finish.

This minimises GPU load operations (one per sweep, not one per condition) and
mirrors the field layout of the Inspect suite so the same downstream analysis
tooling can consume both.

Output layout::

    output_root/{run_name}/
    ├── sweep_config.json               # full RolloutSweepConfig (serialised)
    ├── scale_{s:+.2f}/{condition}/     # one dir per (scale, condition) cell
    │   ├── run_info.json               # scale, condition, aggregates, status
    │   ├── rollouts.jsonl
    │   ├── rollouts_evaluated.jsonl
    │   └── experiment_metadata.json
    └── ...

``scale_+0.00`` is the base model (adapter contribution zeroed).

Usage::

    from scripts.experiments.rollout_experiments.lora_scale_sweep import (
        RolloutSweepConfig, RolloutSweepCondition, ScaleSweep, run_rollout_sweep,
    )

    config = RolloutSweepConfig(
        base_model="meta-llama/Llama-3.1-8B-Instruct",
        adapter="persona-shattering-lasr/o_avoiding_adapter::adapter/final",
        sweep=ScaleSweep(min=-2.0, max=2.0, step=1.0),
        conditions=[
            RolloutSweepCondition(name="no_prompt",   phases=[Phase(num_turns=1)]),
            RolloutSweepCondition(name="o_avoiding",  phases=[Phase(num_turns=1, assistant_system_prompt="...")]),
        ],
        evaluations=["count_o"],
        rollout=RolloutExperimentConfig(...),
        output_root=Path("scratch/runs/my_sweep"),
    )
    run_rollout_sweep(config)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, field_validator, model_validator

from scripts.experiments.rollout_experiments import (
    Phase,
    RolloutExperimentConfig,
    UserSimulatorConfig,
    run_experiment,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class ScaleSweep(BaseModel):
    """LoRA scale sweep parameters.

    Defines a linear grid of adapter scaling factors from *min* to *max*
    (inclusive) at *step* intervals.  0.0 (base model, adapter zeroed) is
    always included automatically and does not need to appear in the grid.

    Example::

        ScaleSweep(min=-2.0, max=2.0, step=1.0)
        # → scale points: [-2.0, -1.0, 0.0, 1.0, 2.0]
    """

    min: float = -2.0
    max: float = 2.0
    step: float = 1.0

    @model_validator(mode="after")
    def _validate_range(self) -> "ScaleSweep":
        if self.step <= 0:
            raise ValueError(f"step must be positive, got {self.step}")
        if self.min > self.max:
            raise ValueError(f"min ({self.min}) must be <= max ({self.max})")
        return self

    def scale_points(self) -> list[float]:
        """Return all scale values including 0.0 (base model)."""
        n_steps = round((self.max - self.min) / self.step)
        points = [round(self.min + i * self.step, 10) for i in range(n_steps + 1)]
        # Ensure 0.0 is always present (base model).
        if 0.0 not in points:
            points.append(0.0)
        return sorted(points)


@dataclass
class RolloutSweepCondition:
    """One condition (system-prompt variant) to run at every scale point.

    Args:
        name: Short identifier used in directory names and ``run_info.json``.
        phases: Phase list passed to ``run_experiment``.
        user_sim: Optional user simulator override for this condition.
    """

    name: str
    phases: list[Phase]
    user_sim: UserSimulatorConfig | None = None


class RolloutSweepConfig(BaseModel):
    """Full configuration for a rollout LoRA scale sweep.

    Mirrors ``SuiteConfig`` in the Inspect eval suite.

    Args:
        base_model: HuggingFace model ID for the base model.
        adapter: Adapter path or ``"repo_id::subfolder"`` reference.
        sweep: Scale grid definition.
        conditions: List of prompt conditions to run at every scale point.
        evaluations: Persona metric names (e.g. ``["count_o"]``).
        rollout: ``RolloutExperimentConfig`` supplying generation settings.
            ``assistant_provider`` must be ``"local"``.
        output_root: Root directory; a timestamped run subdir is created here.
        run_name: Optional fixed run name (timestamped if omitted).
        adapter_name: Internal PEFT adapter name; default matches the Inspect suite.
        dtype: Torch dtype string for model loading.
        skip_completed: Skip (scale, condition) cells that already have a
            ``run_info.json`` with ``status == "ok"``.
        metadata: Arbitrary extra fields written into ``sweep_config.json``.
    """

    model_config = {"arbitrary_types_allowed": True}

    base_model: str
    adapter: str
    sweep: ScaleSweep = ScaleSweep()
    conditions: list[RolloutSweepCondition]
    evaluations: list[str]
    rollout: RolloutExperimentConfig
    output_root: Path
    run_name: str | None = None
    adapter_name: str = "default"
    dtype: str = "bfloat16"
    skip_completed: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @field_validator("rollout")
    @classmethod
    def _require_local(cls, v: RolloutExperimentConfig) -> RolloutExperimentConfig:
        if v.assistant_provider != "local":
            raise ValueError(
                f"rollout.assistant_provider must be 'local', got {v.assistant_provider!r}"
            )
        return v


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, mins = divmod(minutes, 60)
    return f"{hours}h{mins:02d}m{secs:02d}s"


def _scale_label(scale: float) -> str:
    return f"scale_{scale:+.2f}"


def _load_model(
    base_model: str,
    adapter: str,
    adapter_name: str,
    dtype: str,
) -> tuple:
    """Load base model + LoRA adapter once.  Returns ``(peft_model, tokenizer)``."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = getattr(torch, dtype, None)
    if torch_dtype is None:
        raise ValueError(f"Unsupported dtype: {dtype!r}")

    adapter_ref, subfolder = _parse_adapter_ref(adapter)

    print(f"  loading base model: {base_model}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch_dtype, device_map="auto"
    )
    print(
        f"  loading adapter: {adapter_ref}"
        + (f"  (subfolder={subfolder})" if subfolder else ""),
        flush=True,
    )
    peft_kwargs: dict[str, Any] = {"adapter_name": adapter_name}
    if subfolder:
        peft_kwargs["subfolder"] = subfolder
    peft_model = PeftModel.from_pretrained(base, adapter_ref, **peft_kwargs)

    tokenizer_kwargs: dict[str, Any] = {}
    if subfolder:
        tokenizer_kwargs["subfolder"] = subfolder
    tokenizer = AutoTokenizer.from_pretrained(adapter_ref, **tokenizer_kwargs)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_model.config.pad_token_id = tokenizer.pad_token_id
    peft_model.eval()
    return peft_model, tokenizer


def _parse_adapter_ref(adapter: str) -> tuple[str, str | None]:
    """Split ``"repo_id::subfolder"`` → ``(repo_id, subfolder)`` or ``(path, None)``."""
    if "::" in adapter:
        repo, subfolder = adapter.split("::", 1)
        return repo, subfolder
    return adapter, None


def _apply_scale(peft_model: Any, adapter_name: str, scale: float) -> Any:
    """Apply LoRaScaling and return the scaler (caller must call ``.restore()``)."""
    from src.utils.peft_manipulations import LoRaScaling

    return LoRaScaling(peft_model, adapter_name=adapter_name, scale_factor=scale).apply()


def _write_run_info(
    run_dir: Path,
    scale: float,
    condition: str,
    status: str,
    aggregates: dict[str, Any] | None,
    error: str | None,
    elapsed: float | None,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run_info.json"
    path.write_text(
        json.dumps(
            {
                "scale": scale,
                "condition": condition,
                "status": status,
                "aggregates": aggregates,
                "error": error,
                "elapsed_seconds": round(elapsed, 2) if elapsed is not None else None,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return path


def _run_experiment_with_preloaded(
    config: RolloutExperimentConfig,
    name: str,
    phases: list[Phase],
    evaluations: list[str],
    user_sim: UserSimulatorConfig | None,
    peft_model: Any,
    tokenizer: Any,
) -> Any:
    """Run ``run_experiment`` injecting a pre-loaded model via a scoped patch."""
    import scripts.experiments.rollout_experiments as _re_mod

    original_build = _re_mod.build_assistant_inference

    def _patched_build(cfg: RolloutExperimentConfig):
        inf_cfg = original_build(cfg)
        new_local = inf_cfg.local.model_copy(update={"preloaded_model": (peft_model, tokenizer)})
        return inf_cfg.model_copy(update={"local": new_local})

    _re_mod.build_assistant_inference = _patched_build
    try:
        return run_experiment(config, name, phases, evaluations, user_sim=user_sim)
    finally:
        _re_mod.build_assistant_inference = original_build


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_rollout_sweep(config: RolloutSweepConfig) -> Path:
    """Execute a full rollout scale sweep and return the output root directory.

    Loop structure (matches the Inspect suite):
        for scale in scale_points:          ← apply LoRaScaling once
            for condition in conditions:    ← run_experiment per condition
        restore → next scale

    Args:
        config: Full sweep configuration.

    Returns:
        Path to the timestamped run directory containing all results.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = config.run_name or timestamp
    output_root = config.output_root / run_name
    output_root.mkdir(parents=True, exist_ok=True)

    scale_points = config.sweep.scale_points()
    n_scales = len(scale_points)
    n_conditions = len(config.conditions)

    print(
        f"\n=== Rollout sweep: {run_name} "
        f"| {n_scales} scale(s) × {n_conditions} condition(s) ===",
        flush=True,
    )

    # Write config for reproducibility.
    (output_root / "sweep_config.json").write_text(
        json.dumps(
            {
                "base_model": config.base_model,
                "adapter": config.adapter,
                "sweep": config.sweep.model_dump(),
                "conditions": [c.name for c in config.conditions],
                "evaluations": config.evaluations,
                "scale_points": scale_points,
                **config.metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Load model once for the whole sweep.
    load_t0 = time.perf_counter()
    print("  loading model for sweep (once) ...", flush=True)
    peft_model, tokenizer = _load_model(
        config.base_model, config.adapter, config.adapter_name, config.dtype
    )
    print(f"  model loaded  ({_fmt_duration(time.perf_counter() - load_t0)})", flush=True)

    suite_t0 = time.perf_counter()
    timings: list[tuple[str, str, str, float]] = []

    for scale_idx, scale in enumerate(scale_points, 1):
        slabel = _scale_label(scale)
        print(f"\n  [{scale_idx}/{n_scales}] scale={scale:+.3f}", flush=True)

        scaler = _apply_scale(peft_model, config.adapter_name, scale)
        try:
            for condition in config.conditions:
                cell_dir = output_root / slabel / condition.name
                run_info_path = cell_dir / "run_info.json"

                cell_label = f"{slabel}/{condition.name}"

                if config.skip_completed and run_info_path.exists():
                    try:
                        info = json.loads(run_info_path.read_text())
                        if info.get("status") == "ok":
                            print(f"    skipping  {cell_label}  (already done)", flush=True)
                            timings.append((slabel, condition.name, "skipped", 0.0))
                            continue
                    except Exception:
                        pass

                print(f"    running   {cell_label} ...", flush=True)
                cell_t0 = time.perf_counter()

                # Point scratch_dir at the cell dir so run_experiment puts its
                # timestamped subdir there.
                cell_config = RolloutExperimentConfig(
                    **{
                        **vars(config.rollout),
                        "scratch_dir": cell_dir,
                    }
                )

                try:
                    result = _run_experiment_with_preloaded(
                        cell_config,
                        name=condition.name,
                        phases=condition.phases,
                        evaluations=config.evaluations,
                        user_sim=condition.user_sim,
                        peft_model=peft_model,
                        tokenizer=tokenizer,
                    )
                    elapsed = time.perf_counter() - cell_t0
                    _write_run_info(
                        cell_dir, scale, condition.name, "ok",
                        result.aggregates, None, elapsed,
                    )
                    timings.append((slabel, condition.name, "ok", elapsed))
                    print(
                        f"    done      {cell_label}  ({_fmt_duration(elapsed)})",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    elapsed = time.perf_counter() - cell_t0
                    _write_run_info(
                        cell_dir, scale, condition.name, "failed",
                        None, str(exc), elapsed,
                    )
                    timings.append((slabel, condition.name, "failed", elapsed))
                    print(
                        f"    FAILED    {cell_label}  ({_fmt_duration(elapsed)}): {exc}",
                        flush=True,
                    )
        finally:
            scaler.restore()

    # Release GPU memory.
    try:
        peft_model.cpu()
    except Exception:
        pass

    _print_timing_summary(timings, time.perf_counter() - suite_t0)
    return output_root


def _print_timing_summary(
    timings: list[tuple[str, str, str, float]],
    suite_elapsed: float,
) -> None:
    if not timings:
        print(f"\n=== Sweep done in {_fmt_duration(suite_elapsed)} (no cells ran) ===\n", flush=True)
        return

    col_scale = max(max(len(s) for s, _, _, _ in timings), 5)
    col_cond = max(max(len(c) for _, c, _, _ in timings), 9)

    header = f"  {'Scale':<{col_scale}}  {'Condition':<{col_cond}}  {'Status':<7}  Time"
    sep = "  " + "-" * (col_scale + col_cond + 22)
    print("\n=== Timing summary ===", flush=True)
    print(header, flush=True)
    print(sep, flush=True)
    for scale_label, condition, status, elapsed in timings:
        t = _fmt_duration(elapsed) if elapsed > 0 else "-"
        print(f"  {scale_label:<{col_scale}}  {condition:<{col_cond}}  {status:<7}  {t}", flush=True)
    print(sep, flush=True)
    print(f"  Total: {_fmt_duration(suite_elapsed)}", flush=True)
    print("======================\n", flush=True)
