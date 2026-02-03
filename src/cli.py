"""Command-line interface for persona extraction pipeline."""

import subprocess
import sys
from pathlib import Path

import click

from .config import load_config


@click.group()
@click.version_option(version="0.1.0")
def main():
    """Persona extraction via LoRA fine-tuning."""
    pass


# Map of stage names to script modules
STAGE_SCRIPTS = {
    "inference": "scripts.run_inference",
    "editing": "scripts.edit_responses",
    # training and evaluation not yet implemented as scripts
}


@main.command()
@click.argument("config_path", type=click.Path(exists=True))
def run(config_path: str):
    """Run the full pipeline with the given configuration."""
    config = load_config(config_path)
    stages = config.get("stages", [])

    click.echo(f"Running pipeline with config: {config_path}")
    click.echo(f"Stages: {stages}")

    for stage in stages:
        click.echo(f"\n{'='*60}")
        click.echo(f"Starting stage: {stage}")
        click.echo(f"{'='*60}")

        script_module = STAGE_SCRIPTS.get(stage)
        if script_module is None:
            click.echo(f"⚠️  Stage '{stage}' not yet implemented, skipping.", err=True)
            continue

        # Run the script as a module
        result = subprocess.run(
            [sys.executable, "-m", script_module, config_path],
            check=False,
        )

        if result.returncode != 0:
            click.echo(f"❌ Stage '{stage}' failed with exit code {result.returncode}", err=True)
            sys.exit(result.returncode)

        click.echo(f"✓ Stage '{stage}' completed successfully")

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

    script_module = STAGE_SCRIPTS.get(stage_name)
    if script_module is None:
        click.echo(f"❌ Unknown or unimplemented stage: '{stage_name}'", err=True)
        click.echo(f"Available stages: {', '.join(STAGE_SCRIPTS.keys())}", err=True)
        sys.exit(1)

    # Run the script as a module
    result = subprocess.run(
        [sys.executable, "-m", script_module, config_path],
        check=False,
    )

    if result.returncode != 0:
        click.echo(f"❌ Stage '{stage_name}' failed with exit code {result.returncode}", err=True)
        sys.exit(result.returncode)

    click.echo(f"✓ Stage '{stage_name}' completed successfully")


if __name__ == "__main__":
    main()
