"""Tests for the evals module."""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from scripts.evals.cli import main as cli_main
from scripts.evals.config import AdapterConfig, EvalConfig
from scripts.evals.run import _build_model_args, _save_results, run_eval


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestEvalConfig:
    def test_minimal_valid(self):
        cfg = EvalConfig(model="gpt2", tasks=["mmlu"])
        assert cfg.model == "gpt2"
        assert cfg.tasks == ["mmlu"]
        assert cfg.adapters == []
        assert cfg.needs_merge is False

    def test_empty_tasks_rejected(self):
        with pytest.raises(ValueError, match="tasks must not be empty"):
            EvalConfig(model="gpt2", tasks=[])

    def test_single_adapter_no_merge(self):
        cfg = EvalConfig(
            model="gpt2",
            tasks=["mmlu"],
            adapters=[AdapterConfig(path="/tmp/adapter")],
        )
        assert cfg.needs_merge is False

    def test_single_adapter_scaled_needs_merge(self):
        cfg = EvalConfig(
            model="gpt2",
            tasks=["mmlu"],
            adapters=[AdapterConfig(path="/tmp/adapter", scale=0.5)],
        )
        assert cfg.needs_merge is True

    def test_multi_adapter_needs_merge(self):
        cfg = EvalConfig(
            model="gpt2",
            tasks=["mmlu"],
            adapters=[
                AdapterConfig(path="/tmp/a"),
                AdapterConfig(path="/tmp/b"),
            ],
        )
        assert cfg.needs_merge is True

    def test_infinite_scale_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            AdapterConfig(path="/tmp/a", scale=float("inf"))

    def test_nan_scale_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            AdapterConfig(path="/tmp/a", scale=float("nan"))

    def test_negative_scale_allowed(self):
        adapter = AdapterConfig(path="/tmp/a", scale=-1.0)
        assert adapter.scale == -1.0


# ---------------------------------------------------------------------------
# _build_model_args
# ---------------------------------------------------------------------------


class TestBuildModelArgs:
    def test_base_model_only(self):
        cfg = EvalConfig(model="meta-llama/Llama-3.1-8B", tasks=["mmlu"])
        args = _build_model_args(cfg)
        assert args == "pretrained=meta-llama/Llama-3.1-8B"

    def test_with_peft(self):
        cfg = EvalConfig(model="meta-llama/Llama-3.1-8B", tasks=["mmlu"])
        args = _build_model_args(cfg, peft="/tmp/adapter")
        assert "pretrained=meta-llama/Llama-3.1-8B" in args
        assert "peft=/tmp/adapter" in args

    def test_pretrained_override(self):
        cfg = EvalConfig(model="meta-llama/Llama-3.1-8B", tasks=["mmlu"])
        args = _build_model_args(cfg, pretrained_override="/tmp/merged")
        assert "pretrained=/tmp/merged" in args
        assert "meta-llama" not in args

    def test_extra_model_args(self):
        cfg = EvalConfig(
            model="gpt2",
            tasks=["mmlu"],
            model_args={"dtype": "float16", "trust_remote_code": "true"},
        )
        args = _build_model_args(cfg)
        assert "dtype=float16" in args
        assert "trust_remote_code=true" in args


# ---------------------------------------------------------------------------
# run_eval model_args routing
# ---------------------------------------------------------------------------


class TestRunEval:
    @patch("scripts.evals.run.lm_eval.evaluator.simple_evaluate")
    @patch("scripts.evals.run.TaskManager")
    def test_base_model_no_adapters(self, mock_tm, mock_eval):
        mock_eval.return_value = {"results": {}}
        cfg = EvalConfig(model="gpt2", tasks=["mmlu"])

        run_eval(cfg)

        call_kwargs = mock_eval.call_args[1]
        assert "peft" not in call_kwargs["model_args"]
        assert "pretrained=gpt2" in call_kwargs["model_args"]

    @patch("scripts.evals.run.lm_eval.evaluator.simple_evaluate")
    @patch("scripts.evals.run.TaskManager")
    def test_single_adapter_scale_1(self, mock_tm, mock_eval):
        mock_eval.return_value = {"results": {}}
        cfg = EvalConfig(
            model="gpt2",
            tasks=["mmlu"],
            adapters=[AdapterConfig(path="/tmp/adapter")],
        )

        run_eval(cfg)

        call_kwargs = mock_eval.call_args[1]
        assert "peft=/tmp/adapter" in call_kwargs["model_args"]

    @patch("scripts.evals.run.shutil.rmtree")
    @patch("scripts.evals.run.merge_adapters")
    @patch("scripts.evals.run.tempfile.mkdtemp", return_value="/tmp/merged_123")
    @patch("scripts.evals.run.lm_eval.evaluator.simple_evaluate")
    @patch("scripts.evals.run.TaskManager")
    def test_scaled_adapter_merges_and_cleans_up(
        self, mock_tm, mock_eval, mock_mkdtemp, mock_merge, mock_rmtree
    ):
        mock_eval.return_value = {"results": {}}
        cfg = EvalConfig(
            model="gpt2",
            tasks=["mmlu"],
            adapters=[AdapterConfig(path="/tmp/adapter", scale=0.5)],
        )

        run_eval(cfg)

        # Merge was called
        mock_merge.assert_called_once()
        merge_kwargs = mock_merge.call_args[1]
        assert merge_kwargs["base_model"] == "gpt2"
        assert len(merge_kwargs["adapters"]) == 1
        assert merge_kwargs["adapters"][0].scale == 0.5

        # lm_eval was called with the merged path
        call_kwargs = mock_eval.call_args[1]
        assert "pretrained=/tmp/merged_123" in call_kwargs["model_args"]
        assert "peft" not in call_kwargs["model_args"]

        # Temp dir was cleaned up
        mock_rmtree.assert_called_once_with("/tmp/merged_123", ignore_errors=True)

    @patch("scripts.evals.run.shutil.rmtree")
    @patch("scripts.evals.run.merge_adapters")
    @patch("scripts.evals.run.tempfile.mkdtemp", return_value="/tmp/merged_err")
    @patch("scripts.evals.run.lm_eval.evaluator.simple_evaluate", side_effect=RuntimeError("boom"))
    @patch("scripts.evals.run.TaskManager")
    def test_cleanup_on_error(
        self, mock_tm, mock_eval, mock_mkdtemp, mock_merge, mock_rmtree
    ):
        cfg = EvalConfig(
            model="gpt2",
            tasks=["mmlu"],
            adapters=[AdapterConfig(path="/tmp/adapter", scale=0.5)],
        )

        with pytest.raises(RuntimeError, match="boom"):
            run_eval(cfg)

        # Temp dir still cleaned up
        mock_rmtree.assert_called_once_with("/tmp/merged_err", ignore_errors=True)


# ---------------------------------------------------------------------------
# Output saving
# ---------------------------------------------------------------------------


class TestSaveResults:
    def test_saves_results_json(self, tmp_path):
        results = {
            "results": {"mmlu": {"acc": 0.5}},
            "configs": {"mmlu": {}},
            "versions": {"mmlu": 1},
        }
        _save_results(results, tmp_path)

        results_file = tmp_path / "results.json"
        assert results_file.exists()

        import json
        saved = json.loads(results_file.read_text())
        assert saved["results"]["mmlu"]["acc"] == 0.5


class TestCli:
    def test_list_tasks_does_not_require_model_or_tasks(self):
        runner = CliRunner()
        result = runner.invoke(cli_main, ["--list-tasks"])
        assert result.exit_code == 0
        assert "Custom persona metric tasks" in result.output
