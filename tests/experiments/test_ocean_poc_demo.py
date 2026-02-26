"""Tests for OCEAN POC orchestration script."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from datasets import Dataset

from scripts.evals.config import RunSummaryRow, SuiteResult
from scripts.experiments.poc import ocean_poc_demo
from scripts.utils import write_jsonl


def test_ocean_poc_dry_run_train(tmp_path: Path, capsys) -> None:
    run_dir = tmp_path / "poc_run"
    rc = ocean_poc_demo.main(
        [
            "train",
            "--run-dir",
            str(run_dir),
            "--dry-run",
        ]
    )
    assert rc == 0
    output = capsys.readouterr().out
    assert "DRY RUN: training phase plan" in output
    assert "quick-test-openness" in output


def test_ocean_poc_all_smoke_with_patched_backends(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "poc_run"

    def _fake_run_inference(config, dataset=None):
        records = [
            {"question": f"Q{i}", "response": f"A{i}"}
            for i in range(10)
        ]
        if config.output_path is not None:
            write_jsonl(records, config.output_path)
        return Dataset.from_list(records), SimpleNamespace(output_path=config.output_path)

    def _fake_run_editing(config, dataset=None):
        assert dataset is not None
        edited = []
        for row in dataset.to_list():
            edited.append(
                {
                    "question": row["question"],
                    "response": row["response"],
                    "edited_response": f"edited::{row['response']}",
                }
            )
        if config.output_path is not None:
            write_jsonl(edited, config.output_path)
        return Dataset.from_list(edited), SimpleNamespace(num_samples=len(edited), num_failed=0)

    def _fake_run_training(config):
        final_path = config.checkpoint_dir / "final"
        final_path.mkdir(parents=True, exist_ok=True)
        (final_path / "adapter_config.json").write_text("{}", encoding="utf-8")
        return Dataset.from_list([]), SimpleNamespace(
            checkpoint_path=final_path,
            num_train_samples=9,
            num_val_samples=1,
        )

    def _fake_run_suite(config, judge_exec):
        run_name = config.run_name or "run"
        output_root = config.output_root / run_name
        rows = []
        for model in config.models:
            for eval_spec in config.evals:
                out_dir = output_root / model.name / eval_spec.name
                out_dir.mkdir(parents=True, exist_ok=True)
                log_path = out_dir / f"{model.name}_{eval_spec.name}.json"
                log_path.write_text("{}", encoding="utf-8")
                rows.append(
                    RunSummaryRow(
                        model_name=model.base_model,
                        model_spec_name=model.name,
                        eval_name=eval_spec.name,
                        eval_kind=eval_spec.kind,
                        status="ok",
                        output_dir=str(out_dir),
                        inspect_log_path=str(log_path),
                    )
                )
        return SuiteResult(output_root=output_root, rows=rows)

    def _decode_scale_tag(tag: str) -> float:
        if tag.startswith("neg_"):
            return -float(tag[4:].replace("p", "."))
        return float(tag.replace("p", "."))

    def _fake_read_metrics(log_path: str) -> dict[str, float]:
        name = Path(log_path).stem
        if name.endswith("_truthfulqa_mc1"):
            return {"accuracy": 0.55}
        if name.endswith("_gsm8k"):
            return {"accuracy": 0.42}

        model_name = name[: -len("_personality_trait")]
        dims = {dim: 0.0 for dim in ocean_poc_demo.OCEAN_DIMENSIONS}

        if "_s_" in model_name:
            trait, scale_tag = model_name.split("_s_", 1)
            scale = _decode_scale_tag(scale_tag)
            target = ocean_poc_demo.TRAIT_TO_DIMENSION[trait]
            dims[target] = scale * 0.1
        elif model_name in ocean_poc_demo.TRAIT_TO_DIMENSION:
            target = ocean_poc_demo.TRAIT_TO_DIMENSION[model_name]
            dims[target] = 0.25
        elif model_name == "combo":
            dims["Openness"] = -0.10
            dims["Conscientiousness"] = 0.00
            dims["Extraversion"] = 0.07
            dims["Agreeableness"] = 0.10
            dims["Neuroticism"] = 0.05

        return {f"trait_ratio.{key}": value for key, value in dims.items()}

    monkeypatch.setattr(ocean_poc_demo, "run_inference", _fake_run_inference)
    monkeypatch.setattr(ocean_poc_demo, "run_editing", _fake_run_editing)
    monkeypatch.setattr(ocean_poc_demo, "run_training", _fake_run_training)
    monkeypatch.setattr(ocean_poc_demo, "_run_suite", _fake_run_suite)
    monkeypatch.setattr(ocean_poc_demo, "read_numeric_metrics_from_log", _fake_read_metrics)

    rc = ocean_poc_demo.main(
        [
            "all",
            "--run-dir",
            str(run_dir),
            "--train-samples",
            "10",
            "--train-epochs",
            "1",
            "--eval-samples",
            "5",
            "--scales=-1,0,1",
            "--skip-hf-upload",
            "--no-wandb",
        ]
    )

    assert rc == 0
    assert (run_dir / "run_manifest.json").exists()
    assert (run_dir / "README_poc_outputs.md").exists()
    assert (run_dir / "scaling_summary.csv").exists()
    assert (run_dir / "scaling_long.csv").exists()
    assert (run_dir / "model_eval_long.csv").exists()
    assert (run_dir / "model_eval_wide.csv").exists()
    assert (run_dir / "plots" / "figure1_scaling_sweeps.png").exists()
    assert (run_dir / "plots" / "figure2_eval_summary.png").exists()
