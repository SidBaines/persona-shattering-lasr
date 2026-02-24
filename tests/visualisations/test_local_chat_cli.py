"""Tests for browser local chat CLI wrapper and import guard."""

from __future__ import annotations

import pytest

from scripts.visualisations import local_chat
from scripts.visualisations.local_chat_web import app


def test_parse_args_has_expected_browser_defaults() -> None:
    args = local_chat.parse_args([])
    assert args.host == "127.0.0.1"
    assert args.port == 7860
    assert args.inbrowser is False


def test_main_gracefully_reports_launch_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_runtime_error(_args):
        raise RuntimeError("Gradio is required for browser local chat")

    monkeypatch.setattr(local_chat, "_launch_browser_chat", _raise_runtime_error)

    with pytest.raises(SystemExit, match="Gradio is required"):
        local_chat.main([])


def test_gradio_import_guard_mentions_ui_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    original = app.importlib.import_module

    def fake_import_module(name: str):
        if name == "gradio":
            raise ImportError("gradio missing")
        return original(name)

    monkeypatch.setattr(app.importlib, "import_module", fake_import_module)

    with pytest.raises(RuntimeError, match="uv sync --extra ui"):
        app._get_gradio()  # pylint: disable=protected-access
