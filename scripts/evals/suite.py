"""Suite orchestration for Inspect-based eval runs."""

from __future__ import annotations

import importlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from inspect_ai.model import Model, get_model
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.evals.backends.inspect_runner import (
    run_benchmark_eval,
    run_custom_eval,
    score_custom_eval_from_log,
)
from scripts.evals.config import (
    AdapterConfig,
    InspectBenchmarkSpec,
    InspectCustomEvalSpec,
    JudgeExecutionConfig,
    ModelSpec,
    RunSummaryRow,
    SuiteConfig,
    SuiteResult,
)
from scripts.evals.model_resolution import resolve_model_reference
from scripts.evals.utils.preloaded_hf_provider import register_preloaded_hf_provider
from scripts.utils.lora_composition import load_and_scale_adapters
from src.utils.peft_manipulations import LoRaScaling

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, mins = divmod(minutes, 60)
    return f"{hours}h{mins:02d}m{secs:02d}s"


# ---------------------------------------------------------------------------
# GPU / Inspect cache cleanup
# ---------------------------------------------------------------------------

def _cleanup_runtime_model_state() -> None:
    """Release Inspect's in-process model cache and free GPU memory."""
    try:
        from inspect_ai.model import _model as inspect_model_impl

        active_model_cv = getattr(inspect_model_impl, "active_model_context_var", None)
        if active_model_cv is not None and hasattr(active_model_cv, "set"):
            active_model_cv.set(None)

        model_roles_cv = getattr(inspect_model_impl, "_model_roles", None)
        if model_roles_cv is not None and hasattr(model_roles_cv, "set"):
            model_roles_cv.set({})

        cached_models = getattr(inspect_model_impl, "_models", None)
        if isinstance(cached_models, dict):
            for model in list(cached_models.values()):
                api = getattr(model, "api", None)
                # The HF provider's batch thread holds a dangling generator reference
                # that keeps the model on GPU. Moving to CPU before close() frees VRAM.
                hf_model = getattr(api, "model", None)
                if hf_model is not None and callable(getattr(hf_model, "cpu", None)):
                    try:
                        hf_model.cpu()
                    except Exception:
                        pass
                close = getattr(api, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
            cached_models.clear()
    except Exception:
        pass

    try:
        import gc
        gc.collect()
        gc.collect()
    except Exception:
        pass

    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Model preparation — the unified path
# ---------------------------------------------------------------------------

@dataclass
class _PreparedModel:
    """A model ready to be passed to Inspect, plus optional cleanup."""

    # Either a URI string (API / resume) or a live Model object (local HF).
    inspect_model: str | Model
    # Non-None only when LoRaScaling was applied; must be restored after the eval.
    scaler: LoRaScaling | None
    # The underlying PeftModel, kept so we can move it off GPU after the suite.
    peft_model: PeftModel | None
    # Human-readable name for logging.
    model_name: str


def _resolve_dtype(spec: ModelSpec) -> torch.dtype:
    dtype = getattr(torch, spec.dtype, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported dtype: {spec.dtype!r}")
    return dtype


def _load_local_model(spec: ModelSpec, batch_size: int | None) -> _PreparedModel:
    """Load a local HF model (with optional adapters) into GPU memory."""
    register_preloaded_hf_provider()

    base_ref = resolve_model_reference(spec.base_model, kind="base model")
    torch_dtype = _resolve_dtype(spec)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_ref,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    scaler: LoRaScaling | None = None

    if spec.adapters:
        from scripts.utils.lora_composition import normalize_weighted_adapters
        from scripts.evals.model_resolution import resolve_model_reference as _resolve

        normalized = normalize_weighted_adapters(spec.adapters)
        peft_model, adapter_names, _ = load_and_scale_adapters(
            base_model,
            adapters=normalized,
            adapter_name_prefix="adapter",
            adapter_resolver=lambda ref: _resolve(ref, kind="adapter"),
        )
        tokenizer_ref = resolve_model_reference(normalized[0].path, kind="adapter")
    else:
        peft_model = PeftModel.__new__(PeftModel)
        # No adapters — wrap as a plain model; use a lightweight shim instead.
        # Actually for no-adapter case we skip PeftModel entirely.
        peft_model = base_model  # type: ignore[assignment]
        tokenizer_ref = base_ref

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_ref)

    inspect_model = get_model(
        f"hf_preloaded/{spec.name}",
        hf_model=peft_model,
        hf_tokenizer=tokenizer,
        batch_size=batch_size or 32,
    )

    return _PreparedModel(
        inspect_model=inspect_model,
        scaler=scaler,
        peft_model=peft_model,
        model_name=spec.base_model,
    )


def _load_local_model_for_sweep(
    base_model_ref: str,
    adapter_ref: str,
    dtype: torch.dtype,
    subfolder: str | None = None,
) -> tuple[PeftModel, Any]:
    """Load base model + single adapter once for a scale sweep."""
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_ref,
        torch_dtype=dtype,
        device_map="auto",
    )
    peft_kwargs: dict[str, Any] = {}
    if subfolder:
        peft_kwargs["subfolder"] = subfolder
    peft_model = PeftModel.from_pretrained(base_model, adapter_ref, **peft_kwargs)
    # Tokenizer lives in the adapter subfolder if one is specified, otherwise the adapter root.
    tokenizer_kwargs: dict[str, Any] = {}
    if subfolder:
        tokenizer_kwargs["subfolder"] = subfolder
    tokenizer = AutoTokenizer.from_pretrained(adapter_ref, **tokenizer_kwargs)
    return peft_model, tokenizer


def _prepare_sweep_model(
    spec: ModelSpec,
    peft_model: PeftModel,
    tokenizer: Any,
    batch_size: int | None,
) -> _PreparedModel:
    """Wrap a sweep model spec: apply LoRaScaling for this scale point."""
    register_preloaded_hf_provider()

    scaler: LoRaScaling | None = None
    if spec.scale is not None:
        scaler = LoRaScaling(
            peft_model,
            adapter_name="default",
            scale_factor=spec.scale,
        ).apply()

    inspect_model = get_model(
        f"hf_preloaded/{spec.name}",
        hf_model=peft_model,
        hf_tokenizer=tokenizer,
        batch_size=batch_size or 32,
    )
    return _PreparedModel(
        inspect_model=inspect_model,
        scaler=scaler,
        peft_model=peft_model,
        model_name=spec.base_model,
    )


def _prepare_api_model(spec: ModelSpec) -> _PreparedModel:
    """Wrap an API model spec (model_uri already set)."""
    assert spec.model_uri is not None
    return _PreparedModel(
        inspect_model=spec.model_uri,
        scaler=None,
        peft_model=None,
        model_name=spec.base_model,
    )


def _prepare_resume_model(spec: ModelSpec) -> _PreparedModel:
    return _PreparedModel(
        inspect_model=f"hf/{spec.base_model}",
        scaler=None,
        peft_model=None,
        model_name=spec.base_model,
    )


# ---------------------------------------------------------------------------
# Helpers carried over from the original suite
# ---------------------------------------------------------------------------

def load_suite_module(module_path: str) -> tuple[SuiteConfig, JudgeExecutionConfig]:
    """Load SUITE_CONFIG (and optional JUDGE_EXEC_CONFIG) from a module."""
    module = importlib.import_module(module_path)
    if not hasattr(module, "SUITE_CONFIG"):
        raise AttributeError(f"Module '{module_path}' must export SUITE_CONFIG")
    suite_config = getattr(module, "SUITE_CONFIG")
    if not isinstance(suite_config, SuiteConfig):
        raise TypeError(f"SUITE_CONFIG must be SuiteConfig, got {type(suite_config)}")
    judge_exec = getattr(module, "JUDGE_EXEC_CONFIG", JudgeExecutionConfig())
    if not isinstance(judge_exec, JudgeExecutionConfig):
        raise TypeError("JUDGE_EXEC_CONFIG must be JudgeExecutionConfig when provided")
    return suite_config, judge_exec


def _make_output_root(config: SuiteConfig, mode: str) -> Path:
    if mode == "resume":
        if not config.run_name:
            raise ValueError(
                "run_name must be set in SuiteConfig when mode='resume' "
                "so the prior run directory can be located."
            )
        run_name = config.run_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = config.run_name or timestamp
    output_root = config.output_root / run_name
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


def _run_dir_for(
    *,
    output_root: Path,
    model_spec_name: str,
    eval_name: str,
    run_index: int = 0,
) -> Path:
    suffix = f"/run_{run_index:02d}" if run_index > 0 else ""
    run_dir = output_root / model_spec_name / f"{eval_name}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _is_scale_in_eval(
    scale: float | None,
    eval_spec: InspectBenchmarkSpec | InspectCustomEvalSpec,
    suite_sweep: Any,
) -> bool:
    """Return True if *scale* is part of this eval's scale grid."""
    if scale is None:
        return True
    if not isinstance(eval_spec, InspectBenchmarkSpec):
        return True
    effective_sweep = eval_spec.sweep if eval_spec.sweep is not None else suite_sweep
    if effective_sweep is None:
        return True
    return scale in set(effective_sweep.scale_points())


def _ensure_hf_log_repo(hf_log_dir: str) -> None:
    prefix = "hf://datasets/"
    if not hf_log_dir.startswith(prefix):
        return
    parts = hf_log_dir[len(prefix):].split("/")
    if len(parts) < 2:
        return
    repo_id = f"{parts[0]}/{parts[1]}"
    try:
        from huggingface_hub import HfApi
        HfApi().create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
    except Exception:
        pass


def _summary_row(
    *,
    model_name: str,
    model_spec_name: str,
    eval_name: str,
    eval_kind: str,
    status: str,
    output_dir: Path,
    run_info_path: Path | None,
    inspect_log_path: str | None,
    error: str | None = None,
) -> RunSummaryRow:
    return RunSummaryRow(
        model_name=model_name,
        model_spec_name=model_spec_name,
        eval_name=eval_name,
        eval_kind=eval_kind,
        status=status,
        output_dir=str(output_dir),
        run_info_path=str(run_info_path) if run_info_path else None,
        inspect_log_path=inspect_log_path,
        error=error,
    )


def _write_run_info(
    *,
    run_dir: Path,
    output_root: Path,
    model_spec: ModelSpec,
    eval_spec: InspectBenchmarkSpec | InspectCustomEvalSpec,
    judge_exec: JudgeExecutionConfig,
    prepared: _PreparedModel,
    status: str,
    error: str | None,
    inspect_log_path: str | None,
    inspect_status: str | None,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run_info.json"
    payload = {
        "suite_run_name": output_root.name,
        "status": status,
        "error": error,
        "model_spec": model_spec.model_dump(mode="json"),
        "eval_spec": eval_spec.model_dump(mode="json"),
        "judge_execution": judge_exec.model_dump(mode="json"),
        "scale": model_spec.scale,
        "native": {
            "inspect_log_path": inspect_log_path,
            "inspect_status": inspect_status,
        },
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def _record_failed_model_rows(
    *,
    rows: list[RunSummaryRow],
    output_root: Path,
    model_spec: ModelSpec,
    evals: list[InspectBenchmarkSpec | InspectCustomEvalSpec],
    judge_exec: JudgeExecutionConfig,
    prepared: _PreparedModel | None,
    error: str,
) -> None:
    for eval_spec in evals:
        run_dir = _run_dir_for(
            output_root=output_root,
            model_spec_name=model_spec.name,
            eval_name=eval_spec.name,
        )
        eval_kind = "benchmark" if isinstance(eval_spec, InspectBenchmarkSpec) else "custom"
        run_info_path = _write_run_info(
            run_dir=run_dir,
            output_root=output_root,
            model_spec=model_spec,
            eval_spec=eval_spec,
            judge_exec=judge_exec,
            prepared=prepared or _PreparedModel(
                inspect_model="", scaler=None, peft_model=None,
                model_name=model_spec.base_model,
            ),
            status="failed",
            error=error,
            inspect_log_path=None,
            inspect_status=None,
        )
        rows.append(
            _summary_row(
                model_name=model_spec.base_model,
                model_spec_name=model_spec.name,
                eval_name=eval_spec.name,
                eval_kind=eval_kind,
                status="failed",
                output_dir=run_dir,
                run_info_path=run_info_path,
                inspect_log_path=None,
                error=error,
            )
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_eval_suite(
    config: SuiteConfig,
    judge_exec: JudgeExecutionConfig | None = None,
) -> SuiteResult:
    """Run a full eval suite using Inspect for all eval types."""
    judge_exec = judge_exec or JudgeExecutionConfig()
    output_root = _make_output_root(config, judge_exec.mode)

    (output_root / "suite_config.json").write_text(
        config.model_dump_json(indent=2), encoding="utf-8"
    )

    if config.hf_log_dir:
        _ensure_hf_log_repo(config.hf_log_dir)

    rows: list[RunSummaryRow] = []
    models = config.expand_models()
    n_models = len(models)
    n_evals = len(config.evals)

    suite_t0 = time.perf_counter()
    print(
        f"\n=== Suite: {output_root.name} | {n_models} model(s) × {n_evals} eval(s) ===",
        flush=True,
    )
    eval_timings: list[tuple[str, str, str, float]] = []

    # --- Sweep: load the base model + adapter once, reuse across all scale points ---
    is_sweep = config.sweep is not None and judge_exec.mode != "resume"
    sweep_peft_model: PeftModel | None = None
    sweep_tokenizer: Any = None

    if is_sweep and config.adapter is not None:
        load_t0 = time.perf_counter()
        print("  loading model for sweep (once) ...", flush=True)
        try:
            assert config.base_model is not None
            first_spec = models[1] if len(models) > 1 else models[0]
            from scripts.utils.lora_composition import split_adapter_reference
            _raw_adapter, _adapter_subfolder = split_adapter_reference(config.adapter)
            base_ref = resolve_model_reference(config.base_model, kind="base model")
            adapter_ref = resolve_model_reference(_raw_adapter, kind="adapter")
            sweep_peft_model, sweep_tokenizer = _load_local_model_for_sweep(
                base_ref, adapter_ref, _resolve_dtype(first_spec),
                subfolder=_adapter_subfolder,
            )
            print(f"  model loaded  ({_fmt_duration(time.perf_counter() - load_t0)})", flush=True)
        except Exception as exc:
            print(f"  FAILED to load sweep model: {exc}", flush=True)
            is_sweep = False  # fall back to per-spec loading

    # --- Per-model loop ---
    for model_idx, model_spec in enumerate(models, 1):
        model_label = f"[{model_idx}/{n_models}] {model_spec.name}"

        # Prepare the model for this spec.
        try:
            if judge_exec.mode == "resume":
                prepared = _prepare_resume_model(model_spec)
            elif model_spec.model_uri is not None:
                prepared = _prepare_api_model(model_spec)
            elif is_sweep and sweep_peft_model is not None:
                prepared = _prepare_sweep_model(
                    model_spec, sweep_peft_model, sweep_tokenizer, config.batch_size
                )
            else:
                print(f"  loading {model_label} ...", flush=True)
                load_t0 = time.perf_counter()
                prepared = _load_local_model(model_spec, config.batch_size)
                print(f"  loaded  {model_label}  ({_fmt_duration(time.perf_counter() - load_t0)})", flush=True)
        except Exception as exc:
            print(f"  FAILED  {model_label}: {exc}", flush=True)
            _record_failed_model_rows(
                rows=rows,
                output_root=output_root,
                model_spec=model_spec,
                evals=config.evals,
                judge_exec=judge_exec,
                prepared=None,
                error=f"model load failed: {exc}",
            )
            continue

        # Run all evals for this model spec.
        try:
            for eval_spec in config.evals:
                if not _is_scale_in_eval(model_spec.scale, eval_spec, config.sweep):
                    continue

                eval_kind = "benchmark" if isinstance(eval_spec, InspectBenchmarkSpec) else "custom"
                n_runs = eval_spec.n_runs if isinstance(eval_spec, InspectBenchmarkSpec) else 1

                for run_index in range(n_runs):
                    run_dir = _run_dir_for(
                        output_root=output_root,
                        model_spec_name=model_spec.name,
                        eval_name=eval_spec.name,
                        run_index=run_index,
                    )
                    run_label = (
                        f"[{model_idx}/{n_models}] {model_spec.name} / {eval_spec.name}"
                        + (f" run {run_index}" if n_runs > 1 else "")
                    )

                    if config.skip_completed:
                        run_info_path = run_dir / "run_info.json"
                        if run_info_path.exists():
                            try:
                                info = json.loads(run_info_path.read_text())
                                if info.get("status") == "ok":
                                    print(f"  skipping  {run_label}  (already done)", flush=True)
                                    rows.append(_summary_row(
                                        model_name=prepared.model_name,
                                        model_spec_name=model_spec.name,
                                        eval_name=eval_spec.name,
                                        eval_kind=eval_kind,
                                        status="skipped",
                                        output_dir=run_dir,
                                        run_info_path=run_info_path,
                                        inspect_log_path=info.get("native", {}).get("inspect_log_path"),
                                    ))
                                    continue
                            except Exception:
                                pass

                    print(f"  running   {run_label} ...", flush=True)
                    eval_t0 = time.perf_counter()

                    hf_log_dir: str | None = None
                    if config.hf_log_dir:
                        base = config.hf_log_dir.rstrip("/")
                        hf_log_dir = f"{base}/{output_root.name}/{model_spec.name}/{eval_spec.name}"

                    if isinstance(eval_spec, InspectBenchmarkSpec):
                        if judge_exec.mode == "resume":
                            result_status = "skipped"
                            result_error = "resume mode does not apply to benchmark evals"
                            inspect_log_path = None
                            inspect_status = None
                        else:
                            result = run_benchmark_eval(
                                spec=eval_spec,
                                model_uri=prepared.inspect_model,
                                run_dir=run_dir,
                                temperature=config.temperature,
                                hf_log_dir=hf_log_dir,
                            )
                            result_status = result.status
                            result_error = result.error
                            inspect_log_path = result.log.location if result.log else None
                            inspect_status = result.log.status if result.log else None
                    else:
                        if judge_exec.mode == "resume":
                            result = score_custom_eval_from_log(
                                spec=eval_spec,
                                run_dir=run_dir,
                                judge_exec=judge_exec,
                            )
                        else:
                            result = run_custom_eval(
                                spec=eval_spec,
                                model_uri=prepared.inspect_model,
                                run_dir=run_dir,
                                judge_exec=judge_exec,
                                hf_log_dir=hf_log_dir,
                            )
                        result_status = result.status
                        result_error = result.error
                        inspect_log_path = result.log.location if result.log else None
                        inspect_status = result.log.status if result.log else None

                    eval_elapsed = time.perf_counter() - eval_t0
                    print(
                        f"  done      {run_label}  ({_fmt_duration(eval_elapsed)}) [{result_status}]",
                        flush=True,
                    )
                    eval_timings.append((model_spec.name, eval_spec.name, result_status, eval_elapsed))

                    run_info_path = _write_run_info(
                        run_dir=run_dir,
                        output_root=output_root,
                        model_spec=model_spec,
                        eval_spec=eval_spec,
                        judge_exec=judge_exec,
                        prepared=prepared,
                        status=result_status,
                        error=result_error,
                        inspect_log_path=inspect_log_path,
                        inspect_status=inspect_status,
                    )
                    rows.append(_summary_row(
                        model_name=prepared.model_name,
                        model_spec_name=model_spec.name,
                        eval_name=eval_spec.name,
                        eval_kind=eval_kind,
                        status=result_status,
                        output_dir=run_dir,
                        run_info_path=run_info_path,
                        inspect_log_path=inspect_log_path,
                        error=result_error,
                    ))

                # Evict Inspect's model cache between evals in the standard path.
                # In sweep mode the model is ours — don't evict it.
                if sweep_peft_model is None:
                    _cleanup_runtime_model_state()

        finally:
            # Always restore LoRaScaling so weights are clean for the next scale point.
            if prepared.scaler is not None:
                prepared.scaler.restore()
            # Only evict Inspect's model cache for per-spec models (not sweep).
            # In sweep mode the same provider instance is reused across scale points;
            # clearing the cache orphans batched_generate's background thread and hangs.
            if sweep_peft_model is None:
                _cleanup_runtime_model_state()
                if prepared.peft_model is not None:
                    try:
                        prepared.peft_model.cpu()
                    except Exception:
                        pass

    # Release the sweep model after all scale points are done.
    if sweep_peft_model is not None:
        try:
            sweep_peft_model.cpu()
        except Exception:
            pass
        _cleanup_runtime_model_state()

    suite_elapsed = time.perf_counter() - suite_t0
    _print_timing_summary(eval_timings, suite_elapsed)

    return SuiteResult(output_root=output_root, rows=rows)


def _print_timing_summary(
    eval_timings: list[tuple[str, str, str, float]],
    suite_elapsed: float,
) -> None:
    if not eval_timings:
        print(f"\n=== Suite done in {_fmt_duration(suite_elapsed)} (no evals ran) ===\n", flush=True)
        return

    col_model = max(max(len(m) for m, _, _, _ in eval_timings), 5)
    col_eval = max(max(len(e) for _, e, _, _ in eval_timings), 4)

    header = f"  {'Model':<{col_model}}  {'Eval':<{col_eval}}  {'Status':<7}  Time"
    sep = "  " + "-" * (col_model + col_eval + 22)
    print("\n=== Timing summary ===", flush=True)
    print(header, flush=True)
    print(sep, flush=True)
    for model_name, eval_name, status, elapsed in eval_timings:
        print(
            f"  {model_name:<{col_model}}  {eval_name:<{col_eval}}  {status:<7}  {_fmt_duration(elapsed)}",
            flush=True,
        )
    print(sep, flush=True)
    print(f"  Total: {_fmt_duration(suite_elapsed)}", flush=True)
    print("======================\n", flush=True)


def run_inspect_eval(
    *,
    model: Any,
    eval_spec: Any,
    output_root: Path,
    judge_exec: JudgeExecutionConfig | None = None,
    run_name: str | None = None,
) -> SuiteResult:
    """Convenience wrapper for running one model against one eval."""
    config = SuiteConfig(
        models=[model],
        evals=[eval_spec],
        output_root=output_root,
        run_name=run_name,
    )
    return run_eval_suite(config, judge_exec=judge_exec)
