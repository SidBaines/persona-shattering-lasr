"""Suite orchestration for Inspect-based eval runs."""

from __future__ import annotations

import importlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.evals.backends.inspect_runner import (
    run_benchmark_eval,
    run_custom_eval,
    score_custom_eval_from_log,
)
from scripts.evals.config import (
    InspectBenchmarkSpec,
    InspectCustomEvalSpec,
    JudgeExecutionConfig,
    ModelSpec,
    RunSummaryRow,
    SuiteConfig,
    SuiteResult,
)
from scripts.evals.model_materialization import MaterializedModel, materialize_model
from scripts.utils.lora_composition import delete_materialized_model_dir


def _cleanup_runtime_model_state() -> None:
    """Release in-process model state between model specs."""
    try:
        # Inspect memoizes model objects globally; clear them so sequential
        # model specs in one process don't keep prior model weights resident.
        from inspect_ai.model import _model as inspect_model_impl

        # Reset active model / role context vars first so they don't retain refs.
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
                # The Inspect HF provider's background batch thread stores a
                # `generator = functools.partial(self.model.generate, ...)` local
                # variable that is never reassigned after the last batch (because
                # the try-block only runs when len(inputs) > 0).  This reference
                # keeps the AutoModelForCausalLM alive on GPU indefinitely.
                # Moving the model to CPU *before* close() zeroes api.model frees
                # VRAM even while that dangling generator reference persists.
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
        gc.collect()  # Second pass handles any cyclic references revealed by the first.
    except Exception:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for any in-flight CUDA ops to finish.
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass


def load_suite_module(
    module_path: str,
) -> tuple[SuiteConfig, JudgeExecutionConfig]:
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
    # run_index=0 keeps the single-run path unchanged for backwards compat.
    suffix = f"/run_{run_index:02d}" if run_index > 0 else ""
    run_dir = output_root / model_spec_name / f"{eval_name}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _resume_materialized_model(model_spec: ModelSpec) -> MaterializedModel:
    return MaterializedModel(
        model_name=model_spec.base_model,
        model_spec_name=model_spec.name,
        model_uri=f"hf/{model_spec.base_model}",
        cache_key="resume",
        materialized_path=None,
    )


def _materialized_payload(materialized: MaterializedModel | None, model_spec: ModelSpec) -> dict[str, Any]:
    if materialized is None:
        return {
            "model_name": model_spec.base_model,
            "model_spec_name": model_spec.name,
            "model_uri": None,
            "cache_key": None,
            "materialized_path": None,
        }
    return {
        "model_name": materialized.model_name,
        "model_spec_name": materialized.model_spec_name,
        "model_uri": materialized.model_uri,
        "cache_key": materialized.cache_key,
        "materialized_path": (
            str(materialized.materialized_path)
            if materialized.materialized_path
            else None
        ),
    }


def _summary_row_base(
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


def _inspect_model_args(model_spec: ModelSpec) -> dict[str, Any]:
    args: dict[str, Any] = dict(model_spec.inspect_model_args)

    # Inspect's HuggingFace provider already passes `device_map` internally.
    # Passing it again in model_args raises:
    # "from_pretrained() got multiple values for keyword argument 'device_map'"
    if "device_map" in args:
        raise ValueError(
            "inspect_model_args['device_map'] is not supported. "
            "Use inspect_model_args['device'] instead."
        )

    if "torch_dtype" in args and "dtype" not in args:
        args["dtype"] = args.pop("torch_dtype")

    def _resolve_torch_dtype(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        try:
            import torch
        except Exception:
            return value
        dtype = getattr(torch, value, None)
        return dtype if isinstance(dtype, torch.dtype) else value

    args.setdefault("dtype", _resolve_torch_dtype(model_spec.dtype))
    if "dtype" in args:
        args["dtype"] = _resolve_torch_dtype(args["dtype"])

    if model_spec.device_map not in ("", "auto"):
        args.setdefault("device", model_spec.device_map)

    return args


def _write_run_info_for(
    *,
    run_dir: Path,
    output_root: Path,
    model_spec: ModelSpec,
    eval_spec: InspectBenchmarkSpec | InspectCustomEvalSpec,
    judge_exec: JudgeExecutionConfig,
    inspect_model_args: dict[str, Any],
    materialized: MaterializedModel | None,
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
        "inspect_model_args": inspect_model_args,
        "materialized_model": _materialized_payload(materialized, model_spec),
        # Scale stored explicitly so analyzers don't have to parse model names.
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
    materialized: MaterializedModel | None,
    inspect_model_args: dict[str, Any],
    error: str,
) -> None:
    for eval_spec in evals:
        run_dir = _run_dir_for(
            output_root=output_root,
            model_spec_name=model_spec.name,
            eval_name=eval_spec.name,
        )
        eval_kind = "benchmark" if isinstance(eval_spec, InspectBenchmarkSpec) else "custom"
        run_info_path = _write_run_info_for(
            run_dir=run_dir,
            output_root=output_root,
            model_spec=model_spec,
            eval_spec=eval_spec,
            judge_exec=judge_exec,
            inspect_model_args=inspect_model_args,
            materialized=materialized,
            status="failed",
            error=error,
            inspect_log_path=None,
            inspect_status=None,
        )
        rows.append(
            _summary_row_base(
                model_name=(materialized.model_name if materialized else model_spec.base_model),
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


def _cleanup_materialized_model(
    *,
    config: SuiteConfig,
    judge_exec: JudgeExecutionConfig,
    materialized: MaterializedModel,
) -> None:
    merged_path = materialized.materialized_path
    if (
        config.cleanup_materialized_models
        and judge_exec.mode != "resume"
        and merged_path is not None
    ):
        delete_materialized_model_dir(merged_path, prune_empty_parent=True)


def _ensure_hf_log_repo(hf_log_dir: str) -> None:
    """Create the HF Hub dataset repo for log storage if it does not exist."""
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


def _is_scale_in_eval(
    scale: float | None,
    eval_spec: "InspectBenchmarkSpec | InspectCustomEvalSpec",
    suite_sweep: "ScaleSweep | None",
) -> bool:
    """Return True if *scale* is part of this eval's scale grid.

    The base model (scale=None) is always included.
    For per-eval sweep overrides, only run that eval at its own scale points.
    """
    from scripts.evals.config import InspectBenchmarkSpec
    if scale is None:
        return True
    if not isinstance(eval_spec, InspectBenchmarkSpec):
        return True
    effective_sweep = eval_spec.sweep if eval_spec.sweep is not None else suite_sweep
    if effective_sweep is None:
        return True
    return scale in set(effective_sweep.scale_points())


def run_eval_suite(
    config: SuiteConfig,
    judge_exec: JudgeExecutionConfig | None = None,
) -> SuiteResult:
    """Run a full eval suite using Inspect for all eval types."""
    from scripts.evals.config import InspectBenchmarkSpec, ScaleSweep
    judge_exec = judge_exec or JudgeExecutionConfig()
    output_root = _make_output_root(config, judge_exec.mode)

    # Serialize config for replication before any runs start.
    (output_root / "suite_config.json").write_text(
        config.model_dump_json(indent=2), encoding="utf-8"
    )

    if config.hf_log_dir:
        _ensure_hf_log_repo(config.hf_log_dir)

    rows: list[RunSummaryRow] = []
    models = config.expand_models()

    for model_spec in models:
        if judge_exec.mode == "resume":
            materialized = _resume_materialized_model(model_spec)
        elif model_spec.model_uri is not None:
            # API model (e.g. openrouter/..., openai/...) — skip HF resolution
            # and LoRA materialization entirely; pass the URI straight to Inspect.
            materialized = MaterializedModel(
                model_name=model_spec.base_model,
                model_spec_name=model_spec.name,
                model_uri=model_spec.model_uri,
                cache_key="api",
                materialized_path=None,
            )
        else:
            try:
                materialized = materialize_model(model_spec, output_root)
            except Exception as exc:
                _record_failed_model_rows(
                    rows=rows,
                    output_root=output_root,
                    model_spec=model_spec,
                    evals=config.evals,
                    judge_exec=judge_exec,
                    materialized=None,
                    inspect_model_args={},
                    error=f"model materialization failed: {exc}",
                )
                continue

        try:
            inspect_model_args = (
                {}
                if model_spec.model_uri is not None
                else _inspect_model_args(model_spec)
            )
        except Exception as exc:
            _record_failed_model_rows(
                rows=rows,
                output_root=output_root,
                model_spec=model_spec,
                evals=config.evals,
                judge_exec=judge_exec,
                materialized=materialized,
                inspect_model_args={},
                error=f"invalid inspect model args: {exc}",
            )
            _cleanup_runtime_model_state()
            _cleanup_materialized_model(
                config=config,
                judge_exec=judge_exec,
                materialized=materialized,
            )
            continue

        try:
            for eval_spec in config.evals:
                # Skip this eval for scale points not in its grid.
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

                    # Skip completed runs when requested.
                    if config.skip_completed:
                        run_info_path = run_dir / "run_info.json"
                        if run_info_path.exists():
                            import json as _json
                            try:
                                info = _json.loads(run_info_path.read_text())
                                if info.get("status") == "ok":
                                    rows.append(
                                        _summary_row_base(
                                            model_name=(materialized.model_name if materialized else model_spec.base_model),
                                            model_spec_name=model_spec.name,
                                            eval_name=eval_spec.name,
                                            eval_kind=eval_kind,
                                            status="skipped",
                                            output_dir=run_dir,
                                            run_info_path=run_info_path,
                                            inspect_log_path=info.get("native", {}).get("inspect_log_path"),
                                            error=None,
                                        )
                                    )
                                    continue
                            except Exception:
                                pass  # Re-run if run_info is unreadable

                    if isinstance(eval_spec, InspectBenchmarkSpec):
                        if judge_exec.mode == "resume":
                            error = "resume mode does not apply to benchmark evals"
                            run_info_path = _write_run_info_for(
                                run_dir=run_dir,
                                output_root=output_root,
                                model_spec=model_spec,
                                eval_spec=eval_spec,
                                judge_exec=judge_exec,
                                inspect_model_args=inspect_model_args,
                                materialized=materialized,
                                status="skipped",
                                error=error,
                                inspect_log_path=None,
                                inspect_status=None,
                            )
                            rows.append(
                                _summary_row_base(
                                    model_name=materialized.model_name,
                                    model_spec_name=model_spec.name,
                                    eval_name=eval_spec.name,
                                    eval_kind=eval_kind,
                                    status="skipped",
                                    output_dir=run_dir,
                                    run_info_path=run_info_path,
                                    inspect_log_path=None,
                                    error=error,
                                )
                            )
                            continue

                        hf_eval_log_dir: str | None = None
                        if config.hf_log_dir:
                            base = config.hf_log_dir.rstrip("/")
                            hf_eval_log_dir = f"{base}/{output_root.name}/{model_spec.name}/{eval_spec.name}"

                        result = run_benchmark_eval(
                            spec=eval_spec,
                            model_uri=materialized.model_uri,
                            run_dir=run_dir,
                            inspect_model_args=inspect_model_args,
                            temperature=config.temperature,
                            hf_log_dir=hf_eval_log_dir,
                        )
                    else:
                        if judge_exec.mode == "resume":
                            result = score_custom_eval_from_log(
                                spec=eval_spec,
                                run_dir=run_dir,
                                judge_exec=judge_exec,
                            )
                        else:
                            hf_eval_log_dir_custom: str | None = None
                            if config.hf_log_dir:
                                base = config.hf_log_dir.rstrip("/")
                                hf_eval_log_dir_custom = f"{base}/{output_root.name}/{model_spec.name}/{eval_spec.name}"
                            result = run_custom_eval(
                                spec=eval_spec,
                                model_uri=materialized.model_uri,
                                run_dir=run_dir,
                                judge_exec=judge_exec,
                                inspect_model_args=inspect_model_args,
                                hf_log_dir=hf_eval_log_dir_custom,
                            )

                    inspect_log_path = result.log.location if result.log is not None else None
                    inspect_status = result.log.status if result.log is not None else None
                    run_info_path = _write_run_info_for(
                        run_dir=run_dir,
                        output_root=output_root,
                        model_spec=model_spec,
                        eval_spec=eval_spec,
                        judge_exec=judge_exec,
                        inspect_model_args=inspect_model_args,
                        materialized=materialized,
                        status=result.status,
                        error=result.error,
                        inspect_log_path=inspect_log_path,
                        inspect_status=inspect_status,
                    )

                    rows.append(
                        _summary_row_base(
                            model_name=materialized.model_name,
                            model_spec_name=model_spec.name,
                            eval_name=eval_spec.name,
                            eval_kind=eval_kind,
                            status=result.status,
                            output_dir=run_dir,
                            run_info_path=run_info_path,
                            inspect_log_path=inspect_log_path,
                            error=result.error,
                        )
                    )

                # Inspect reloads the HF model from disk for each task, so the
                # previous task's weights stay resident unless we explicitly
                # evict them before the next task starts.
                _cleanup_runtime_model_state()
        finally:
            _cleanup_runtime_model_state()
            _cleanup_materialized_model(
                config=config,
                judge_exec=judge_exec,
                materialized=materialized,
            )

    return SuiteResult(
        output_root=output_root,
        rows=rows,
    )


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
