"""Command-line interface for persona extraction pipeline."""

import subprocess
import sys

import click

from src.config import load_config, PipelineConfig


# Tracks which implementations exist in each source (src/ vs scripts/)
# Update this when adding new implementations
MODULE_AVAILABILITY: dict[str, dict[str, bool]] = {
    "inference": {"scripts": True, "src": False},
    "editing": {"scripts": True, "src": False},
    "training": {"scripts": True, "src": False},  # Added training stage
    "evaluation": {"scripts": False, "src": False},
}


def resolve_stage_module(config: PipelineConfig, stage: str) -> str:
    """Resolve which module to use for a stage based on config.

    Args:
        config: Pipeline configuration with module_source settings.
        stage: Name of the stage (inference, editing, training, evaluation).

    Returns:
        Module path to run (e.g., "scripts.inference" or "src.inference").

    Raises:
        NotImplementedError: If the requested source doesn't have an implementation.
        ValueError: If the stage is not recognized.
    """
    if stage not in MODULE_AVAILABILITY:
        raise ValueError(
            f"Unknown stage: '{stage}'. "
            f"Available stages: {list(MODULE_AVAILABILITY.keys())}"
        )

    # Get per-stage override, or fall back to global default
    stage_config = getattr(config.module_source, stage, None)
    if stage_config and stage_config.source:
        source = stage_config.source
    else:
        source = config.module_source.default

    # Check if implementation exists
    if not MODULE_AVAILABILITY[stage].get(source, False):
        available = [s for s, v in MODULE_AVAILABILITY[stage].items() if v]
        raise NotImplementedError(
            f"Stage '{stage}' is not implemented in '{source}'. "
            f"Available sources: {available or ['none']}"
        )

    return f"{source}.{stage}"


@click.group()
@click.version_option(version="0.1.0")
def main():
    """Persona extraction via LoRA fine-tuning."""
    pass


@main.command()
@click.argument("config_path", type=click.Path(exists=True))
def run(config_path: str):
    """Run the full pipeline with the given configuration."""
    config = load_config(config_path)
    stages = config.stages

    click.echo(f"Running pipeline with config: {config_path}")
    click.echo(f"Stages: {stages}")
    click.echo(f"Module source default: {config.module_source.default}")

    for stage_name in stages:
        click.echo(f"\n{'='*60}")
        click.echo(f"Starting stage: {stage_name}")
        click.echo(f"{'='*60}")

        try:
            module_path = resolve_stage_module(config, stage_name)
        except (NotImplementedError, ValueError) as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)

        click.echo(f"Using module: {module_path}")

        # Run the script as a module
        result = subprocess.run(
            [sys.executable, "-m", module_path, config_path],
            check=False,
        )

        if result.returncode != 0:
            click.echo(f"❌ Stage '{stage_name}' failed with exit code {result.returncode}", err=True)
            sys.exit(result.returncode)

        click.echo(f"✓ Stage '{stage_name}' completed successfully")

    click.echo(f"\n{'='*60}")
    click.echo("Pipeline completed!")
    click.echo(f"{'='*60}")


@main.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.argument("stage_name", type=str)
def stage(config_path: str, stage_name: str):
    """Run a single pipeline stage."""
    config = load_config(config_path)
    click.echo(f"Running stage '{stage_name}' with config: {config_path}")

    try:
        module_path = resolve_stage_module(config, stage_name)
    except (NotImplementedError, ValueError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    click.echo(f"Using module: {module_path}")

    # Run the script as a module
    result = subprocess.run(
        [sys.executable, "-m", module_path, config_path],
        check=False,
    )

    if result.returncode != 0:
        click.echo(f"❌ Stage '{stage_name}' failed with exit code {result.returncode}", err=True)
        sys.exit(result.returncode)

    click.echo(f"✓ Stage '{stage_name}' completed successfully")


if __name__ == "__main__":
    main()
