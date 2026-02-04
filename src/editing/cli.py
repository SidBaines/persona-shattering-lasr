# ABOUTME: Provides a CLI entry point for the editing stage.
# ABOUTME: Reports stub status until implementations exist.
"""Editing stage CLI stub."""

import sys

import click


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
def main(config_path: str) -> None:
    """Describe the editing stage stub behavior."""
    click.echo("Editing in src is a stub.")
    click.echo("Implement editors in scripts/ or add src/editing implementations.")
    click.echo(f"Config: {config_path}")
    sys.exit(1)


if __name__ == "__main__":
    main()
