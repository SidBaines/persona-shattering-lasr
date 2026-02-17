"""CLI entry point for the evals module."""

from __future__ import annotations

import sys

import click

from scripts.evals.config import AdapterConfig, EvalConfig
from scripts.evals.tasks import list_custom_tasks
from scripts.utils import setup_logging


def _parse_adapters(value: str) -> list[AdapterConfig]:
    """Parse a comma-separated list of ``path:scale`` pairs.

    Scale defaults to 1.0 when omitted (e.g. ``path_a`` = ``path_a:1.0``).
    """
    adapters = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            path, scale_str = item.rsplit(":", 1)
            try:
                scale = float(scale_str)
            except ValueError:
                # Treat as a path that contains colons (e.g. Windows drive)
                path = item
                scale = 1.0
        else:
            path = item
            scale = 1.0
        adapters.append(AdapterConfig(path=path, scale=scale))
    return adapters


@click.command("evals")
@click.option(
    "--model", required=True,
    help="HuggingFace model name or local path.",
)
@click.option(
    "--adapters", default=None,
    help=(
        "Comma-separated adapter path:scale pairs. "
        "Scale defaults to 1.0 if omitted. "
        "Examples: 'path_a', 'path_a:0.7,path_b:0.3'"
    ),
)
@click.option(
    "--tasks", required=True,
    help="Comma-separated task names (standard or custom).",
)
@click.option("--batch-size", default="auto", help="Batch size: 'auto', 'auto:N', or int.")
@click.option("--device", default=None, help="Device, e.g. 'cuda:0'.")
@click.option("--limit", default=None, type=int, help="Max samples per task.")
@click.option("--num-fewshot", default=None, type=int, help="Number of few-shot examples.")
@click.option("--output-path", default=None, type=click.Path(), help="Output directory.")
@click.option("--log-samples/--no-log-samples", default=True, help="Log per-sample results.")
@click.option("--dtype", default="bfloat16", help="Torch dtype.")
@click.option("--max-gen-toks", default=256, type=int, help="Max generation tokens.")
@click.option("--temperature", default=0.0, type=float, help="Generation temperature.")
@click.option("--apply-chat-template", is_flag=True, default=False, help="Apply chat template.")
@click.option("--list-tasks", "show_tasks", is_flag=True, default=False, help="List custom tasks and exit.")
def main(
    model: str,
    adapters: str | None,
    tasks: str,
    batch_size: str,
    device: str | None,
    limit: int | None,
    num_fewshot: int | None,
    output_path: str | None,
    log_samples: bool,
    dtype: str,
    max_gen_toks: int,
    temperature: float,
    apply_chat_template: bool,
    show_tasks: bool,
) -> None:
    """Run evaluations using lm-evaluation-harness."""
    if show_tasks:
        list_custom_tasks()
        sys.exit(0)

    setup_logging()

    # Parse adapters
    adapter_configs = _parse_adapters(adapters) if adapters else []

    # Build model_args dict
    model_args: dict[str, str] = {}
    if dtype:
        model_args["dtype"] = dtype

    from pathlib import Path

    config = EvalConfig(
        model=model,
        adapters=adapter_configs,
        model_args=model_args,
        tasks=[t.strip() for t in tasks.split(",") if t.strip()],
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        device=device,
        limit=limit,
        max_gen_toks=max_gen_toks,
        temperature=temperature,
        apply_chat_template=apply_chat_template,
        output_path=Path(output_path) if output_path else None,
        log_samples=log_samples,
    )

    from scripts.evals.run import run_eval

    results = run_eval(config)

    # Print summary
    if results and "results" in results:
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        import json

        print(json.dumps(results["results"], indent=2, default=str))
