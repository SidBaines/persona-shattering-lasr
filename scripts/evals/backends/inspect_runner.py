"""Inspect backend runner for benchmark and custom eval specifications."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from inspect_ai.log import EvalLog

from scripts.evals.config import (
    InspectBenchmarkSpec,
    InspectCustomEvalSpec,
    JudgeExecutionConfig,
)
from scripts.evals.inspect_benchmarks import build_benchmark_task
from scripts.evals.inspect_custom import build_custom_scorer, build_custom_task
from scripts.evals.judge_orchestration import (
    resume_from_manifest,
    run_task_with_mode,
    write_jobs_manifest,
)


@dataclass
class InspectRunResult:
    status: str
    log: EvalLog | None = None
    error: str | None = None
    manifest_path: Path | None = None


def run_benchmark_eval(
    *,
    spec: InspectBenchmarkSpec,
    model_uri: str,
    run_dir: Path,
    inspect_model_args: dict | None = None,
    hf_log_dir: str | None = None,
) -> InspectRunResult:
    native_log_dir = run_dir / "native" / "inspect_logs"

    try:
        task = build_benchmark_task(spec)
        log = run_task_with_mode(
            task=task,
            model_uri=model_uri,
            native_log_dir=native_log_dir,
            mode="blocking",
            limit=spec.limit,
            judge_exec=JudgeExecutionConfig(mode="blocking", prefer_batch=False),
            inspect_model_args=inspect_model_args,
            log_dir=hf_log_dir,
            generation_args=spec.generation_args,
        )
        return InspectRunResult(status="ok", log=log)
    except Exception as exc:
        return InspectRunResult(status="failed", error=str(exc))


def run_custom_eval(
    *,
    spec: InspectCustomEvalSpec,
    model_uri: str,
    run_dir: Path,
    judge_exec: JudgeExecutionConfig,
    inspect_model_args: dict | None = None,
    hf_log_dir: str | None = None,
) -> InspectRunResult:
    native_log_dir = run_dir / "native" / "inspect_logs"

    try:
        task, scorer_name = build_custom_task(spec)
        log = run_task_with_mode(
            task=task,
            model_uri=model_uri,
            native_log_dir=native_log_dir,
            mode=judge_exec.mode,
            limit=spec.dataset.max_samples,
            judge_exec=judge_exec,
            inspect_model_args=inspect_model_args,
            log_dir=hf_log_dir,
        )

        if judge_exec.mode == "submit":
            manifest_path = write_jobs_manifest(
                run_dir=run_dir,
                log_path=log.location,
                scorer_names=[scorer_name],
                eval_name=spec.name,
            )
            return InspectRunResult(
                status="pending",
                log=log,
                manifest_path=manifest_path,
            )

        return InspectRunResult(status="ok", log=log)

    except Exception as exc:
        return InspectRunResult(status="failed", error=str(exc))


def score_custom_eval_from_log(
    *,
    spec: InspectCustomEvalSpec,
    run_dir: Path,
    judge_exec: JudgeExecutionConfig | None = None,
) -> InspectRunResult:
    manifest_path = run_dir / "jobs" / "manifest.json"

    try:
        scorer_obj, _ = build_custom_scorer(spec)
        scored_log = resume_from_manifest(
            manifest_path=manifest_path,
            scorers=[scorer_obj],
            judge_exec=judge_exec,
        )
        return InspectRunResult(status="ok", log=scored_log, manifest_path=manifest_path)
    except Exception as exc:
        return InspectRunResult(status="failed", error=str(exc), manifest_path=manifest_path)
