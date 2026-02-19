"""Suite orchestration for Inspect-based eval runs."""

from __future__ import annotations

import importlib
import shutil
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
from scripts.evals.output_schema import (
    write_failed_run_summary,
    write_run_config,
    write_run_outputs,
    write_suite_manifest,
    write_suite_summary,
)


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
) -> Path:
    run_dir = output_root / model_spec_name / eval_name
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


def _summary_row_base(
    *,
    model_name: str,
    model_spec_name: str,
    eval_name: str,
    eval_kind: str,
    status: str,
    output_dir: Path,
    error: str | None = None,
    summary_path: Path | None = None,
) -> RunSummaryRow:
    return RunSummaryRow(
        model_name=model_name,
        model_spec_name=model_spec_name,
        eval_name=eval_name,
        eval_kind=eval_kind,
        status=status,
        output_dir=str(output_dir),
        error=error,
        summary_path=str(summary_path) if summary_path else None,
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

    if "dtype" in args and "torch_dtype" not in args:
        args["torch_dtype"] = args.pop("dtype")

    def _resolve_torch_dtype(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        try:
            import torch
        except Exception:
            return value
        dtype = getattr(torch, value, None)
        return dtype if isinstance(dtype, torch.dtype) else value

    args.setdefault("torch_dtype", _resolve_torch_dtype(model_spec.dtype))
    if "torch_dtype" in args:
        args["torch_dtype"] = _resolve_torch_dtype(args["torch_dtype"])

    if model_spec.device_map not in ("", "auto"):
        args.setdefault("device", model_spec.device_map)

    return args


def _write_run_config_for(
    *,
    run_dir: Path,
    output_root: Path,
    model_spec: ModelSpec,
    eval_spec: InspectBenchmarkSpec | InspectCustomEvalSpec,
    judge_exec: JudgeExecutionConfig,
    materialized: MaterializedModel,
    inspect_model_args: dict[str, Any],
) -> None:
    write_run_config(
        run_dir=run_dir,
        suite_run_name=output_root.name,
        model_spec=model_spec.model_dump(mode="json"),
        eval_spec=eval_spec.model_dump(mode="json"),
        judge_execution=judge_exec.model_dump(mode="json"),
        inspect_model_args=inspect_model_args,
        materialized_model={
            "model_name": materialized.model_name,
            "model_spec_name": materialized.model_spec_name,
            "model_uri": materialized.model_uri,
            "cache_key": materialized.cache_key,
            "materialized_path": (
                str(materialized.materialized_path)
                if materialized.materialized_path
                else None
            ),
        },
    )


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

        if materialized is not None:
            _write_run_config_for(
                run_dir=run_dir,
                output_root=output_root,
                model_spec=model_spec,
                eval_spec=eval_spec,
                judge_exec=judge_exec,
                materialized=materialized,
                inspect_model_args=inspect_model_args,
            )
        else:
            write_run_config(
                run_dir=run_dir,
                suite_run_name=output_root.name,
                model_spec=model_spec.model_dump(mode="json"),
                eval_spec=eval_spec.model_dump(mode="json"),
                judge_execution=judge_exec.model_dump(mode="json"),
                inspect_model_args=inspect_model_args,
                materialized_model={
                    "model_name": model_spec.base_model,
                    "model_spec_name": model_spec.name,
                    "model_uri": None,
                    "cache_key": None,
                    "materialized_path": None,
                },
            )

        summary_path = write_failed_run_summary(
            run_dir=run_dir,
            backend="inspect",
            model_name=(materialized.model_name if materialized else model_spec.base_model),
            model_spec_name=model_spec.name,
            eval_name=eval_spec.name,
            eval_kind=eval_kind,
            error=error,
        )
        rows.append(
            _summary_row_base(
                model_name=(materialized.model_name if materialized else model_spec.base_model),
                model_spec_name=model_spec.name,
                eval_name=eval_spec.name,
                eval_kind=eval_kind,
                status="failed",
                output_dir=run_dir,
                error=error,
                summary_path=summary_path,
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
        shutil.rmtree(merged_path, ignore_errors=True)
        parent = merged_path.parent
        if parent.exists():
            try:
                next(parent.iterdir())
            except StopIteration:
                parent.rmdir()


def run_eval_suite(
    config: SuiteConfig,
    judge_exec: JudgeExecutionConfig | None = None,
) -> SuiteResult:
    """Run a full eval suite using Inspect for all eval types."""
    judge_exec = judge_exec or JudgeExecutionConfig()
    output_root = _make_output_root(config, judge_exec.mode)
    rows: list[RunSummaryRow] = []

    for model_spec in config.models:
        if judge_exec.mode == "resume":
            materialized = _resume_materialized_model(model_spec)
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
            inspect_model_args = _inspect_model_args(model_spec)
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
            _cleanup_materialized_model(
                config=config,
                judge_exec=judge_exec,
                materialized=materialized,
            )
            continue

        try:
            for eval_spec in config.evals:
                run_dir = _run_dir_for(
                    output_root=output_root,
                    model_spec_name=model_spec.name,
                    eval_name=eval_spec.name,
                )
                _write_run_config_for(
                    run_dir=run_dir,
                    output_root=output_root,
                    model_spec=model_spec,
                    eval_spec=eval_spec,
                    judge_exec=judge_exec,
                    materialized=materialized,
                    inspect_model_args=inspect_model_args,
                )

                if isinstance(eval_spec, InspectBenchmarkSpec):
                    if judge_exec.mode == "resume":
                        rows.append(
                            _summary_row_base(
                                model_name=materialized.model_name,
                                model_spec_name=model_spec.name,
                                eval_name=eval_spec.name,
                                eval_kind="benchmark",
                                status="skipped",
                                output_dir=run_dir,
                                error="resume mode does not apply to benchmark evals",
                            )
                        )
                        continue

                    result = run_benchmark_eval(
                        spec=eval_spec,
                        model_uri=materialized.model_uri,
                        run_dir=run_dir,
                        inspect_model_args=inspect_model_args,
                    )

                    if result.log is not None:
                        summary_path, _, _ = write_run_outputs(
                            run_dir=run_dir,
                            log=result.log,
                            backend="inspect",
                            model_name=materialized.model_name,
                            model_spec_name=model_spec.name,
                            eval_name=eval_spec.name,
                            eval_kind="benchmark",
                            status=result.status,
                            metrics_key="persona_metrics",
                            error=result.error,
                        )
                    else:
                        summary_path = write_failed_run_summary(
                            run_dir=run_dir,
                            backend="inspect",
                            model_name=materialized.model_name,
                            model_spec_name=model_spec.name,
                            eval_name=eval_spec.name,
                            eval_kind="benchmark",
                            error=result.error or "run failed before inspect log creation",
                        )

                    rows.append(
                        _summary_row_base(
                            model_name=materialized.model_name,
                            model_spec_name=model_spec.name,
                            eval_name=eval_spec.name,
                            eval_kind="benchmark",
                            status=result.status if result.status != "pending" else "ok",
                            output_dir=run_dir,
                            error=result.error,
                            summary_path=summary_path,
                        )
                    )
                    continue

                if judge_exec.mode == "resume":
                    result = score_custom_eval_from_log(
                        spec=eval_spec,
                        run_dir=run_dir,
                        judge_exec=judge_exec,
                    )
                else:
                    result = run_custom_eval(
                        spec=eval_spec,
                        model_uri=materialized.model_uri,
                        run_dir=run_dir,
                        judge_exec=judge_exec,
                        inspect_model_args=inspect_model_args,
                    )

                if result.log is not None:
                    summary_path, _, _ = write_run_outputs(
                        run_dir=run_dir,
                        log=result.log,
                        backend="inspect",
                        model_name=materialized.model_name,
                        model_spec_name=model_spec.name,
                        eval_name=eval_spec.name,
                        eval_kind="custom",
                        status=result.status,
                        metrics_key=eval_spec.metrics_key,
                        error=result.error,
                    )
                else:
                    summary_path = write_failed_run_summary(
                        run_dir=run_dir,
                        backend="inspect",
                        model_name=materialized.model_name,
                        model_spec_name=model_spec.name,
                        eval_name=eval_spec.name,
                        eval_kind="custom",
                        error=result.error or "run failed before inspect log creation",
                    )

                rows.append(
                    _summary_row_base(
                        model_name=materialized.model_name,
                        model_spec_name=model_spec.name,
                        eval_name=eval_spec.name,
                        eval_kind="custom",
                        status=result.status,
                        output_dir=run_dir,
                        error=result.error,
                        summary_path=summary_path,
                    )
                )
        finally:
            _cleanup_materialized_model(
                config=config,
                judge_exec=judge_exec,
                materialized=materialized,
            )

    manifest_path = write_suite_manifest(
        output_root=output_root,
        manifest={
            "generated_at": datetime.now().isoformat(),
            "suite_config": config.model_dump(mode="json"),
            "judge_execution": judge_exec.model_dump(mode="json"),
            "rows": [row.model_dump(mode="json") for row in rows],
        },
    )
    summary_path = write_suite_summary(
        output_root=output_root,
        rows=[row.model_dump(mode="json") for row in rows],
    )

    return SuiteResult(
        output_root=output_root,
        suite_manifest_path=manifest_path,
        suite_summary_path=summary_path,
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
