"""Tests for Inspect-based eval suite modules."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from scripts.common.config import DatasetConfig, GenerationConfig
from scripts.evals.cli import main as cli_main
from scripts.evals.config import (
    AdapterConfig,
    InspectBenchmarkSpec,
    InspectCustomEvalSpec,
    JudgeExecutionConfig,
    ModelSpec,
    SuiteConfig,
)
from scripts.evals.model_materialization import materialize_model
from scripts.evals.output_schema import write_run_outputs
from scripts.evals.run import run_eval
from scripts.evals.suite import load_suite_module, run_eval_suite


class TestMigrationShim:
    def test_run_eval_raises_migration_error(self):
        with pytest.raises(RuntimeError, match="deprecated"):
            run_eval({})


class TestSuiteConfig:
    def test_union_specs_parse(self, tmp_path: Path):
        cfg = SuiteConfig(
            output_root=tmp_path,
            models=[
                ModelSpec(name="base", base_model="meta-llama/Llama-3.1-8B-Instruct")
            ],
            evals=[
                InspectBenchmarkSpec(name="mmlu", benchmark="mmlu", limit=10),
                InspectCustomEvalSpec(
                    name="custom",
                    dataset=DatasetConfig(source="local", path="scratch/in.jsonl"),
                    input_builder="scripts.evals.examples:question_input_builder",
                    evaluations=["count_o"],
                    generation=GenerationConfig(max_new_tokens=64),
                ),
            ],
        )
        assert len(cfg.models) == 1
        assert len(cfg.evals) == 2

    def test_empty_models_rejected(self, tmp_path: Path):
        with pytest.raises(ValueError, match="models must not be empty"):
            SuiteConfig(output_root=tmp_path, models=[], evals=[])


class TestConfigModuleLoader:
    def test_load_suite_module(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        module_file = tmp_path / "suite_module_test.py"
        module_file.write_text(
            """
from pathlib import Path
from scripts.evals import SuiteConfig, ModelSpec, InspectBenchmarkSpec
SUITE_CONFIG = SuiteConfig(
    output_root=Path('scratch/evals'),
    models=[ModelSpec(name='base', base_model='meta-llama/Llama-3.1-8B-Instruct')],
    evals=[InspectBenchmarkSpec(name='mmlu', benchmark='mmlu')],
)
""",
            encoding="utf-8",
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        if "suite_module_test" in sys.modules:
            del sys.modules["suite_module_test"]

        config, judge = load_suite_module("suite_module_test")
        assert config.models[0].name == "base"
        assert judge.mode == "blocking"


class TestModelMaterialization:
    def test_base_model_uri(self, tmp_path: Path):
        model = ModelSpec(name="base", base_model="meta-llama/Llama-3.1-8B-Instruct")
        result = materialize_model(model, tmp_path)
        assert result.materialized_path is None
        assert result.model_uri.startswith("hf/")

    def test_adapter_merge_cached(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        calls: list[dict] = []

        def fake_merge_adapters(**kwargs):
            calls.append(kwargs)
            out = kwargs["output_dir"]
            out.mkdir(parents=True, exist_ok=True)
            (out / "config.json").write_text("{}", encoding="utf-8")
            return out

        monkeypatch.setattr(
            "scripts.evals.model_materialization.merge_adapters",
            fake_merge_adapters,
        )

        model = ModelSpec(
            name="scaled",
            base_model="meta-llama/Llama-3.1-8B-Instruct",
            adapters=[AdapterConfig(path="/tmp/adapter", scale=2.0)],
        )

        first = materialize_model(model, tmp_path)
        second = materialize_model(model, tmp_path)

        assert first.materialized_path is not None
        assert second.materialized_path == first.materialized_path
        assert len(calls) == 1

    def test_adapter_merge_cached_across_run_dirs(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        calls: list[dict] = []

        def fake_merge_adapters(**kwargs):
            calls.append(kwargs)
            out = kwargs["output_dir"]
            out.mkdir(parents=True, exist_ok=True)
            (out / "config.json").write_text("{}", encoding="utf-8")
            return out

        monkeypatch.setattr(
            "scripts.evals.model_materialization.merge_adapters",
            fake_merge_adapters,
        )

        model = ModelSpec(
            name="scaled",
            base_model="meta-llama/Llama-3.1-8B-Instruct",
            adapters=[AdapterConfig(path="/tmp/adapter", scale=2.0)],
        )

        run_dir_1 = tmp_path / "suite" / "20260218_100000"
        run_dir_2 = tmp_path / "suite" / "20260218_110000"
        first = materialize_model(model, run_dir_1)
        second = materialize_model(model, run_dir_2)

        assert first.materialized_path is not None
        assert second.materialized_path == first.materialized_path
        assert len(calls) == 1

    def test_partial_merge_cleanup_on_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        def fake_merge_adapters(**kwargs):
            out = kwargs["output_dir"]
            out.mkdir(parents=True, exist_ok=True)
            (out / "partial.bin").write_bytes(b"x")
            raise RuntimeError("disk full")

        monkeypatch.setattr(
            "scripts.evals.model_materialization.merge_adapters",
            fake_merge_adapters,
        )

        model = ModelSpec(
            name="scaled",
            base_model="meta-llama/Llama-3.1-8B-Instruct",
            adapters=[AdapterConfig(path="/tmp/adapter", scale=2.0)],
        )

        output_root = tmp_path / "suite" / "20260218_120000"
        with pytest.raises(RuntimeError, match="disk full"):
            materialize_model(model, output_root)

        cache_root = output_root.parent / "_models_cache"
        assert cache_root.exists()
        assert not any(cache_root.iterdir())


class TestOutputSchema:
    def test_write_run_outputs_custom(self, tmp_path: Path):
        log = SimpleNamespace(
            location=str(tmp_path / "native" / "inspect_logs" / "log.json"),
            status="success",
            eval=SimpleNamespace(task="my-task", model="hf/my-model"),
            results=SimpleNamespace(
                scores=[
                    SimpleNamespace(
                        name="main",
                        metrics={
                            "mean": SimpleNamespace(value=0.5),
                        },
                    )
                ],
                total_samples=1,
                completed_samples=1,
            ),
            samples=[
                SimpleNamespace(
                    id=1,
                    input="What?",
                    target="",
                    output=SimpleNamespace(completion="Answer"),
                    scores={
                        "persona": SimpleNamespace(
                            value=1.0,
                            metadata={
                                "persona_metrics": {
                                    "count_o.count": 2,
                                    "count_o.density": 0.1,
                                }
                            },
                        )
                    },
                    metadata={"question": "What?"},
                )
            ],
        )

        summary_path, records_path, summary = write_run_outputs(
            run_dir=tmp_path,
            log=log,
            backend="inspect",
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            model_spec_name="base",
            eval_name="custom_eval",
            eval_kind="custom",
            status="ok",
            metrics_key="persona_metrics",
        )

        assert summary_path.exists()
        assert records_path.exists()
        assert "count_o.count.mean" in summary["metrics"]


class TestInspectRunnerModes:
    def test_benchmark_unknown_returns_failed(self, tmp_path: Path):
        from scripts.evals.backends.inspect_runner import run_benchmark_eval

        result = run_benchmark_eval(
            spec=InspectBenchmarkSpec(name="bad", benchmark="not_real"),
            model_uri="hf/model",
            run_dir=tmp_path,
            inspect_model_args={},
        )
        assert result.status == "failed"
        assert result.error is not None

    def test_submit_writes_manifest(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from scripts.evals.backends.inspect_runner import run_custom_eval

        fake_log = SimpleNamespace(location=str(tmp_path / "native" / "inspect_logs" / "log.json"))

        monkeypatch.setattr(
            "scripts.evals.backends.inspect_runner.build_custom_task",
            lambda spec: (SimpleNamespace(scorer=["s"]), "scorer_a"),
        )
        monkeypatch.setattr(
            "scripts.evals.backends.inspect_runner.run_task_with_mode",
            lambda **kwargs: fake_log,
        )

        spec = InspectCustomEvalSpec(
            name="custom",
            dataset=DatasetConfig(source="local", path="scratch/in.jsonl"),
            input_builder="scripts.evals.examples:question_input_builder",
            evaluations=["count_o"],
            generation=GenerationConfig(max_new_tokens=64),
        )

        result = run_custom_eval(
            spec=spec,
            model_uri="hf/model",
            run_dir=tmp_path,
            judge_exec=JudgeExecutionConfig(mode="submit"),
            inspect_model_args={},
        )

        assert result.status == "pending"
        assert result.manifest_path is not None
        assert result.manifest_path.exists()

    def test_resume_calls_score(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from scripts.evals.backends.inspect_runner import score_custom_eval_from_log

        manifest = tmp_path / "jobs" / "manifest.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            json.dumps({"log_path": str(tmp_path / "native" / "inspect_logs" / "log.json")}),
            encoding="utf-8",
        )

        monkeypatch.setattr(
            "scripts.evals.backends.inspect_runner.build_custom_scorer",
            lambda spec: ("s", "scorer_a"),
        )
        monkeypatch.setattr(
            "scripts.evals.backends.inspect_runner.resume_from_manifest",
            lambda **kwargs: SimpleNamespace(),
        )

        spec = InspectCustomEvalSpec(
            name="custom",
            dataset=DatasetConfig(source="local", path="scratch/in.jsonl"),
            input_builder="scripts.evals.examples:question_input_builder",
            evaluations=["count_o"],
            generation=GenerationConfig(max_new_tokens=64),
        )

        result = score_custom_eval_from_log(spec=spec, run_dir=tmp_path)
        assert result.status == "ok"


class TestSuiteFailureHandling:
    def test_unknown_benchmark_does_not_abort_suite(self, tmp_path: Path):
        cfg = SuiteConfig(
            output_root=tmp_path,
            run_name="run",
            models=[ModelSpec(name="base", base_model="hf/model")],
            evals=[InspectBenchmarkSpec(name="bad", benchmark="not_real")],
        )
        result = run_eval_suite(cfg)

        assert len(result.rows) == 1
        row = result.rows[0]
        assert row.status == "failed"
        assert row.summary_path is not None
        assert Path(row.summary_path).exists()


class TestSuiteModelArgs:
    def test_dtype_and_device_map_passed_to_inspect(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        from scripts.evals.model_materialization import MaterializedModel

        captured: dict[str, object] = {}

        monkeypatch.setattr(
            "scripts.evals.suite.materialize_model",
            lambda model, output_root: MaterializedModel(
                model_name=model.base_model,
                model_spec_name=model.name,
                model_uri=f"hf/{model.base_model}",
                cache_key="k",
                materialized_path=None,
            ),
        )

        def fake_run_benchmark_eval(**kwargs):
            captured["inspect_model_args"] = kwargs["inspect_model_args"]
            return SimpleNamespace(status="failed", log=None, error="boom")

        monkeypatch.setattr("scripts.evals.suite.run_benchmark_eval", fake_run_benchmark_eval)

        cfg = SuiteConfig(
            output_root=tmp_path,
            run_name="r",
            models=[
                ModelSpec(
                    name="m",
                    base_model="hf/model",
                    dtype="float16",
                    device_map="cpu",
                    inspect_model_args={
                        "trust_remote_code": True,
                        "dtype": "bfloat16",
                    },
                )
            ],
            evals=[InspectBenchmarkSpec(name="mmlu", benchmark="mmlu")],
        )
        run_eval_suite(cfg)

        inspect_model_args = captured["inspect_model_args"]
        assert isinstance(inspect_model_args, dict)
        assert inspect_model_args["device_map"] == "cpu"
        assert inspect_model_args["dtype"] == "bfloat16"
        assert inspect_model_args["trust_remote_code"] is True


class TestSuiteMaterializedCleanup:
    def test_merged_model_deleted_after_model_evals(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        from scripts.evals.model_materialization import MaterializedModel

        merged_dir = tmp_path / "merged" / "abc123"

        def fake_materialize_model(model, output_root):
            merged_dir.mkdir(parents=True, exist_ok=True)
            (merged_dir / "config.json").write_text("{}", encoding="utf-8")
            return MaterializedModel(
                model_name=model.base_model,
                model_spec_name=model.name,
                model_uri=f"hf/{merged_dir}",
                cache_key="abc123",
                materialized_path=merged_dir,
            )

        monkeypatch.setattr("scripts.evals.suite.materialize_model", fake_materialize_model)
        monkeypatch.setattr(
            "scripts.evals.suite.run_benchmark_eval",
            lambda **kwargs: SimpleNamespace(status="failed", log=None, error="boom"),
        )

        cfg = SuiteConfig(
            output_root=tmp_path / "suite_out",
            run_name="run",
            models=[
                ModelSpec(
                    name="scaled",
                    base_model="meta-llama/Llama-3.1-8B-Instruct",
                    adapters=[AdapterConfig(path="/tmp/adapter", scale=1.0)],
                )
            ],
            evals=[InspectBenchmarkSpec(name="truthfulqa", benchmark="truthfulqa")],
        )
        run_eval_suite(cfg)

        assert not merged_dir.exists()


class TestCliMigration:
    def test_old_flags_rejected(self):
        runner = CliRunner()
        result = runner.invoke(cli_main, ["--model", "gpt2", "suite"])
        assert result.exit_code != 0
        assert "deprecated" in result.output.lower()
