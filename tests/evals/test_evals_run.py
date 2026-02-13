"""Tests for end-to-end eval runner."""

from datasets import Dataset

from scripts.common.config import GenerationConfig
from scripts.evals import (
    EvalModelConfig,
    EvalsConfig,
    InspectTaskSuiteConfig,
    PersonaMetricsSuiteConfig,
)
from scripts.evals.run import run_evals


def test_run_evals_persona_metrics_suite(tmp_path, monkeypatch):
    prompts = Dataset.from_list([{"question": "Q1"}, {"question": "Q2"}])
    generated = Dataset.from_list(
        [
            {"question": "Q1", "response": "Hello world", "response_index": 0},
            {"question": "Q2", "response": "Sky", "response_index": 0},
        ]
    )

    monkeypatch.setattr(
        "scripts.evals.run._generate_responses_for_model",
        lambda model_cfg, dataset, evals_config: generated,
    )
    class _DummyResult:
        num_samples = 2
        aggregates = {"count_o.count.mean": 1.0}

    monkeypatch.setattr(
        "scripts.evals.run.run_persona_metrics",
        lambda metrics_config, dataset: (
            Dataset.from_list(
                [
                    {
                        "question": "Q1",
                        "response": "Hello world",
                        "response_index": 0,
                        "persona_metrics": {"count_o.count": 2},
                    },
                    {
                        "question": "Q2",
                        "response": "Sky",
                        "response_index": 0,
                        "persona_metrics": {"count_o.count": 0},
                    },
                ]
            ),
            _DummyResult(),
        ),
    )

    config = EvalsConfig(
        models=[EvalModelConfig(kind="base", model="dummy/model")],
        suites=[PersonaMetricsSuiteConfig(evaluations=["count_o"])],
        output_dir=tmp_path / "evals",
        generation=GenerationConfig(
            max_new_tokens=16,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
            batch_size=2,
            num_responses_per_prompt=1,
        ),
    )

    out_dataset, result = run_evals(config, dataset=prompts)

    assert result.num_models == 1
    assert result.num_suites == 1
    assert len(out_dataset) == 2
    assert (tmp_path / "evals" / "leaderboard.json").exists()
    assert (tmp_path / "evals" / "summary.json").exists()
    model_dirs = [path for path in (tmp_path / "evals").iterdir() if path.is_dir()]
    assert model_dirs
    model_dir = model_dirs[0]
    assert (model_dir / "persona_metrics" / "responses.jsonl").exists()
    assert (model_dir / "persona_metrics" / "scored.jsonl").exists()
    assert (model_dir / "persona_metrics" / "suite_result.json").exists()


def test_run_evals_inspect_task_suite(tmp_path, monkeypatch):
    prompts = Dataset.from_list([{"question": "Q1"}])
    generated = Dataset.from_list(
        [{"question": "Q1", "response": "Hello", "response_index": 0}]
    )

    monkeypatch.setattr(
        "scripts.evals.run._generate_responses_for_model",
        lambda model_cfg, dataset, evals_config: generated,
    )
    monkeypatch.setattr(
        "scripts.evals.run._run_inspect_eval",
        lambda task, model_ref, task_params: {"accuracy": 0.75, "meta": {"count": 4}},
    )

    config = EvalsConfig(
        models=[EvalModelConfig(kind="base", model="dummy/model")],
        suites=[InspectTaskSuiteConfig(task="mmlu")],
        output_dir=tmp_path / "evals",
    )

    out_dataset, result = run_evals(config, dataset=prompts)

    assert result.num_models == 1
    assert result.num_suites == 1
    assert len(out_dataset) == 1
    assert result.leaderboard
    model_row = result.leaderboard[0]
    assert "inspect.mmlu.accuracy" in model_row
