"""Tests for the inference CLI stub."""

from click.testing import CliRunner

from src.inference.cli import main


def test_cli_stub_outputs_message_and_exits_nonzero(tmp_path):
    """CLI stub should print status and exit with non-zero code."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(main, [str(config_path)])

    assert result.exit_code == 1
    assert "Inference in src is a stub." in result.output
    assert "Implement providers in scripts/" in result.output
    assert f"Config: {config_path}" in result.output
