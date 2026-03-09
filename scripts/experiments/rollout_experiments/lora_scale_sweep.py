"""LoRA scale sweep over multi-phase rollout experiments.

Provides ``RolloutScaleSweep`` — a thin wrapper around ``run_experiment`` that:
  1. Loads a base model + LoRA adapter once.
  2. Iterates over a grid of scaling factors via ``LoRaScaling`` (in-place,
     reversible).
  3. Runs ``run_experiment`` at each scale point using the pre-loaded model via
     ``LocalProviderConfig.preloaded_model``.
  4. Writes ``sweep_results.json`` into ``scratch_dir`` with per-scale aggregates
     and run-dir paths — mirroring the ``run_info.json`` lineage pattern from the
     Inspect eval sweep so the same downstream analysis tooling can consume both.

Usage::

    from scripts.experiments.rollout_experiments.lora_scale_sweep import (
        RolloutScaleSweep, ScalePoint,
    )

    sweep = RolloutScaleSweep(
        config=CONFIG,
        base_model="meta-llama/Llama-3.1-8B-Instruct",
        adapter_path="persona-shattering-lasr/o_avoiding_adapter::adapter/final",
        scale_points=[ScalePoint(-1.0), ScalePoint(0.0), ScalePoint(0.5), ScalePoint(1.0)],
        experiment_name="o_avoiding_lora_sweep",
    )
    sweep.run(phases, evaluations=["count_o"])

``ScalePoint(0.0)`` corresponds to the base model (adapter zeroed out).
Negative scales invert the LoRA direction.

The output directory layout mirrors the rollout experiment layout::

    scratch_dir/
    └── {experiment_name}_{timestamp}/
        ├── sweep_results.json          # per-scale aggregates + run dirs
        ├── scale_0.00/                 # base model
        │   ├── rollouts.jsonl
        │   ├── rollouts_evaluated.jsonl
        │   └── experiment_metadata.json
        ├── scale_-1.00/
        └── scale_1.00/
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.experiments.rollout_experiments import (
    Phase,
    RolloutExperimentConfig,
    UserSimulatorConfig,
    run_experiment,
)
from scripts.inference.config import LocalProviderConfig


@dataclass
class ScalePoint:
    """One point in a LoRA scale sweep.

    Args:
        scale: Multiplicative scale applied to the adapter contribution.
            0.0 = base model (adapter zeroed out).
            1.0 = adapter at trained strength.
            Negative values invert the LoRA direction.
    """

    scale: float


@dataclass
class RolloutScaleSweep:
    """Runs a rollout experiment at multiple LoRA scales with a single model load.

    Args:
        config: Experiment configuration.  ``assistant_provider`` must be
            ``"local"``; the sweep manages model loading and injection.
        base_model: HuggingFace model ID for the base model.
        adapter_path: Path or HuggingFace repo + subfolder for the LoRA adapter.
            Subfolder syntax: ``"repo_id::subfolder"``.
        scale_points: Ordered list of scale points to sweep.
        experiment_name: Human-readable name used for the sweep output directory
            and as the ``name`` prefix passed to ``run_experiment``.
        adapter_name: PEFT adapter name to use when loading.  Defaults to
            ``"sweep_adapter"`` (the same convention as the Inspect suite).
        dtype: torch dtype string for model loading (e.g. ``"bfloat16"``).
    """

    config: RolloutExperimentConfig
    base_model: str
    adapter_path: str
    scale_points: list[ScalePoint]
    experiment_name: str
    adapter_name: str = "sweep_adapter"
    dtype: str = "bfloat16"

    def run(
        self,
        phases: list[Phase],
        evaluations: list[str],
        *,
        user_sim: UserSimulatorConfig | None = None,
    ) -> dict[str, Any]:
        """Execute the sweep and return a summary dict.

        Args:
            phases: Phase list passed to ``run_experiment`` at each scale point.
            evaluations: Persona metric names (e.g. ``["count_o"]``).
            user_sim: Optional default user simulator override.

        Returns:
            Sweep summary dict written to ``sweep_results.json``.
        """
        if self.config.assistant_provider != "local":
            raise ValueError(
                "RolloutScaleSweep requires assistant_provider='local' "
                f"(got {self.config.assistant_provider!r}). "
                "The sweep manages model loading; remote providers are not supported."
            )

        peft_model, tokenizer = self._load_model()

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        sweep_dir = self.config.scratch_dir / f"{self.experiment_name}_{timestamp}"
        sweep_dir.mkdir(parents=True, exist_ok=True)

        results: list[dict[str, Any]] = []
        for sp in self.scale_points:
            scale_label = f"scale_{sp.scale:+.2f}"
            print(f"\n{'=' * 60}")
            print(f"Scale sweep: {self.experiment_name} | scale={sp.scale:+.3f}")
            print(f"{'=' * 60}")

            scaler = self._apply_scale(peft_model, sp.scale)
            try:
                # Point the experiment's scratch_dir at the per-scale subdir so
                # run_experiment creates its timestamped run dir there.
                scale_config = dataclasses.replace(
                    self.config,
                    scratch_dir=sweep_dir / scale_label,
                    assistant_provider="local",
                )
                # Inject the pre-loaded model via LocalProviderConfig.
                # build_assistant_inference inside run_phased_rollout will use
                # config.assistant_provider/model, but we need to smuggle the
                # pre-loaded object in.  We patch the config's local sub-config
                # via a one-shot wrapper that overrides build_assistant_inference.
                result = _run_experiment_with_preloaded(
                    scale_config,
                    name=scale_label,
                    phases=phases,
                    evaluations=evaluations,
                    user_sim=user_sim,
                    peft_model=peft_model,
                    tokenizer=tokenizer,
                )
                # Find the run directory that was just created.
                run_dirs = sorted((sweep_dir / scale_label).iterdir()) if (sweep_dir / scale_label).exists() else []
                run_dir_path = str(run_dirs[-1]) if run_dirs else None
                results.append(
                    {
                        "scale": sp.scale,
                        "scale_label": scale_label,
                        "run_dir": run_dir_path,
                        "aggregates": result.aggregates,
                        "num_conversations": result.num_conversations,
                        "num_messages_evaluated": result.num_messages_evaluated,
                        "status": "ok",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  Scale {sp.scale}: FAILED — {exc}")
                results.append(
                    {
                        "scale": sp.scale,
                        "scale_label": scale_label,
                        "run_dir": None,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
            finally:
                scaler.restore()

        summary = {
            "experiment_name": self.experiment_name,
            "base_model": self.base_model,
            "adapter_path": self.adapter_path,
            "sweep_dir": str(sweep_dir),
            "scale_points": [sp.scale for sp in self.scale_points],
            "results": results,
        }
        (sweep_dir / "sweep_results.json").write_text(
            json.dumps(summary, indent=2, default=str)
        )
        print(f"\nSweep complete. Results in {sweep_dir}/sweep_results.json")
        return summary

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_model(self) -> tuple:
        """Load base model + LoRA adapter once.  Returns (peft_model, tokenizer)."""
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = getattr(torch, self.dtype, None)
        if dtype is None:
            raise ValueError(f"Unsupported dtype: {self.dtype!r}")

        # Support "repo_id::subfolder" syntax (same as the Inspect suite).
        adapter_ref, subfolder = _parse_adapter_path(self.adapter_path)

        print(f"\nLoading base model: {self.base_model}")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=dtype,
            device_map="auto",
        )
        print(f"Loading LoRA adapter: {adapter_ref}" + (f" (subfolder={subfolder})" if subfolder else ""))
        peft_kwargs: dict[str, Any] = {"adapter_name": self.adapter_name}
        if subfolder:
            peft_kwargs["subfolder"] = subfolder
        peft_model = PeftModel.from_pretrained(base_model, adapter_ref, **peft_kwargs)

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

    def _apply_scale(self, peft_model: Any, scale: float) -> Any:
        """Apply LoRaScaling for this scale point and return the scaler (for restore)."""
        from src.utils.peft_manipulations import LoRaScaling

        # scale=0.0 → base model (adapter zeroed out); matches Inspect sweep convention.
        scaler = LoRaScaling(
            peft_model,
            adapter_name=self.adapter_name,
            scale_factor=scale,
        ).apply()
        return scaler


def _parse_adapter_path(adapter_path: str) -> tuple[str, str | None]:
    """Split ``"repo_id::subfolder"`` into ``(repo_id, subfolder)`` or ``(path, None)``."""
    if "::" in adapter_path:
        repo, subfolder = adapter_path.split("::", 1)
        return repo, subfolder
    return adapter_path, None


def _run_experiment_with_preloaded(
    config: RolloutExperimentConfig,
    name: str,
    phases: list[Phase],
    evaluations: list[str],
    *,
    user_sim: UserSimulatorConfig | None,
    peft_model: Any,
    tokenizer: Any,
) -> Any:
    """Run ``run_experiment`` with a pre-loaded model injected via LocalProviderConfig.

    Monkey-patches ``build_assistant_inference`` in the rollout_experiments module
    so that the ``LocalProvider`` receives the already-in-memory model instead of
    loading from disk.  The patch is scoped to this call.
    """
    import scripts.experiments.rollout_experiments as _re_mod

    original_build = _re_mod.build_assistant_inference

    def _patched_build(cfg: RolloutExperimentConfig):
        inf_cfg = original_build(cfg)
        # Replace the local sub-config with one that carries the preloaded model.
        new_local = LocalProviderConfig(
            **{
                k: v
                for k, v in inf_cfg.local.model_dump().items()
                if k != "preloaded_model"
            },
            preloaded_model=(peft_model, tokenizer),
        )
        return inf_cfg.model_copy(update={"local": new_local})

    _re_mod.build_assistant_inference = _patched_build
    try:
        return run_experiment(config, name, phases, evaluations, user_sim=user_sim)
    finally:
        _re_mod.build_assistant_inference = original_build
