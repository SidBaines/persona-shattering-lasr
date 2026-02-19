"""CLI entry points for all-Inspect eval suite execution."""

from __future__ import annotations

import click

from scripts.evals.config import JudgeExecutionConfig, SuiteConfig
from scripts.evals.suite import load_suite_module, run_eval_suite
from scripts.utils import setup_logging


def _with_filters(
    config: SuiteConfig,
    *,
    model_filters: tuple[str, ...],
    eval_filters: tuple[str, ...],
) -> SuiteConfig:
    models = config.models
    evals = config.evals

    if model_filters:
        allowed = set(model_filters)
        models = [model for model in config.models if model.name in allowed]
    if eval_filters:
        allowed = set(eval_filters)
        evals = [eval_spec for eval_spec in config.evals if eval_spec.name in allowed]

    return SuiteConfig(
        models=models,
        evals=evals,
        output_root=config.output_root,
        run_name=config.run_name,
        metadata=config.metadata,
    )


@click.group(invoke_without_command=True)
@click.option("--model", default=None, help="[Deprecated] Former lm_eval option.")
@click.option("--tasks", default=None, help="[Deprecated] Former lm_eval option.")
@click.option("--adapters", default=None, help="[Deprecated] Former lm_eval option.")
@click.pass_context
def main(
    ctx: click.Context,
    model: str | None,
    tasks: str | None,
    adapters: str | None,
) -> None:
    """Run evals using Inspect-based suite commands."""
    if model or tasks or adapters:
        raise click.UsageError(
            "lm_eval CLI flags are deprecated. "
            "Use `python -m scripts.evals suite --config-module <module>`"
        )

    if ctx.invoked_subcommand is None:
        raise click.UsageError(
            "Missing command. Use one of: `suite`, `run`."
        )


@main.command("suite")
@click.option(
    "--config-module",
    required=True,
    help="Python module path exporting SUITE_CONFIG.",
)
@click.option(
    "--mode",
    type=click.Choice(["blocking", "submit", "resume"]),
    default=None,
    help="Override judge execution mode.",
)
@click.option(
    "--prefer-batch/--no-prefer-batch",
    default=None,
    help="Override judge batch preference.",
)
def run_suite_command(
    config_module: str,
    mode: str | None,
    prefer_batch: bool | None,
) -> None:
    setup_logging()

    config, judge_exec = load_suite_module(config_module)
    if mode is not None:
        judge_exec.mode = mode
    if prefer_batch is not None:
        judge_exec.prefer_batch = prefer_batch

    result = run_eval_suite(config, judge_exec)
    print(f"Suite output root: {result.output_root}")
    print(f"Suite summary: {result.suite_summary_path}")
    print(f"Suite manifest: {result.suite_manifest_path}")


@main.command("run")
@click.option(
    "--config-module",
    required=True,
    help="Python module path exporting SUITE_CONFIG.",
)
@click.option(
    "--model-name",
    "model_names",
    multiple=True,
    help="Optional model spec name filter (repeatable).",
)
@click.option(
    "--eval-name",
    "eval_names",
    multiple=True,
    help="Optional eval spec name filter (repeatable).",
)
@click.option(
    "--mode",
    type=click.Choice(["blocking", "submit", "resume"]),
    default="blocking",
    help="Judge execution mode.",
)
def run_filtered_command(
    config_module: str,
    model_names: tuple[str, ...],
    eval_names: tuple[str, ...],
    mode: str,
) -> None:
    setup_logging()

    config, judge_exec = load_suite_module(config_module)
    judge_exec = JudgeExecutionConfig(**judge_exec.model_dump())
    judge_exec.mode = mode

    filtered = _with_filters(
        config,
        model_filters=model_names,
        eval_filters=eval_names,
    )
    if not filtered.models:
        raise click.UsageError("No models selected after --model-name filtering")
    if not filtered.evals:
        raise click.UsageError("No evals selected after --eval-name filtering")

    result = run_eval_suite(filtered, judge_exec)
    print(f"Suite output root: {result.output_root}")
    print(f"Suite summary: {result.suite_summary_path}")


if __name__ == "__main__":
    main()
