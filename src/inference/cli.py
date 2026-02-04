# ABOUTME: Provides a CLI entry point for the inference stage.
# ABOUTME: Reports stub status until implementations exist.
"""Inference stage CLI stub."""

import sys

import click


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
def main(config_path: str) -> None:
    """Describe the inference stage stub behavior."""
    click.echo("Inference in src is a stub.")
    click.echo("Implement providers in scripts/ or add src/inference implementations.")
    click.echo(f"Config: {config_path}")
    sys.exit(1)


if __name__ == "__main__":
    main()
