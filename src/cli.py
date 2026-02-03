"""Command-line interface for persona extraction pipeline."""

import click

from .config import load_config


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
    click.echo(f"Running pipeline with config: {config_path}")
    click.echo(f"Stages: {config.get('pipeline', {}).get('stages', [])}")
    raise NotImplementedError("Pipeline runner not yet implemented")


@main.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.argument("stage", type=str)
def stage(config_path: str, stage: str):
    """Run a single pipeline stage."""
    config = load_config(config_path)
    click.echo(f"Running stage '{stage}' with config: {config_path}")
    raise NotImplementedError("Stage runner not yet implemented")


if __name__ == "__main__":
    main()
