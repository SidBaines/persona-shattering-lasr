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


def _mock_persona_inspect_payloads():
    """Shared mock inspect payloads for persona metrics tests."""
    return [
        {
            "eval": {"task": "persona_metrics"},
            "results": {
                "scores": [
                    {
                        "name": "count_o.count",
                        "scorer": "persona_metrics",
                        "reducer": None,
                        "metrics": {"mean": {"value": 1.0}},
                    }
                ]
            },
            "samples": [
                {
                    "metadata": {
                        "record": {
                            "question": "Q1",
                            "response": "Hello world",
                            "response_index": 0,
                        }
                    },
                    "scores": {
                        "persona_metrics": {
                            "value": {"count_o.count": 2},
                            "metadata": {"persona_metrics": {"count_o.count": 2}},
                        }
                    },
                },
                {
                    "metadata": {
                        "record": {
                            "question": "Q2",
                            "response": "Sky",
                            "response_index": 0,
                        }
                    },
                    "scores": {
                        "persona_metrics": {
                            "value": {"count_o.count": 0},
                            "metadata": {"persona_metrics": {"count_o.count": 0}},
                        }
                    },
                },
            ],
        }
    ]


def test_run_evals_persona_metrics_native_path(tmp_path, monkeypatch):
    """Base model uses native path: inspect-ai drives generation."""
    prompts = Dataset.from_list([{"question": "Q1"}, {"question": "Q2"}])

    monkeypatch.setattr(
        "scripts.evals.run.build_native_persona_inspect_task",
        lambda dataset, metrics_config, generation_config, scorer_name="persona_metrics": object(),
    )
    monkeypatch.setattr(
        "scripts.evals.run.normalize_inspect_model_ref",
        lambda model_cfg: "hf/dummy/model",
    )
    monkeypatch.setattr(
        "scripts.evals.run.run_inspect_eval",
        lambda tasks, model_ref, eval_kwargs, log_dir: _mock_persona_inspect_payloads(),
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
    suite_dirs = [path for path in model_dir.iterdir() if path.is_dir()]
    assert suite_dirs
    persona_dirs = [path for path in suite_dirs if path.name.startswith("persona_metrics__")]
    assert persona_dirs
    persona_dir = persona_dirs[0]
    # Native path does not write responses.jsonl (inspect generates directly)
    assert not (persona_dir / "responses.jsonl").exists()
    assert (persona_dir / "scored.jsonl").exists()
    assert (persona_dir / "suite_result.json").exists()


def test_run_evals_persona_metrics_replay_path(tmp_path, monkeypatch):
    """LoRA model uses replay path: custom pipeline generates, inspect scores."""
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
    monkeypatch.setattr(
        "scripts.evals.run.build_replay_persona_inspect_task",
        lambda dataset, metrics_config, scorer_name="persona_metrics": object(),
    )
    monkeypatch.setattr(
        "scripts.evals.run.run_inspect_eval",
        lambda tasks, model_ref, eval_kwargs, log_dir: _mock_persona_inspect_payloads(),
    )

    config = EvalsConfig(
        models=[
            EvalModelConfig(
                kind="lora",
                model="dummy/model",
                adapter_path="/tmp/adapter",
            )
        ],
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
    model_dirs = [path for path in (tmp_path / "evals").iterdir() if path.is_dir()]
    assert model_dirs
    model_dir = model_dirs[0]
    suite_dirs = [path for path in model_dir.iterdir() if path.is_dir()]
    persona_dirs = [path for path in suite_dirs if path.name.startswith("persona_metrics__")]
    assert persona_dirs
    persona_dir = persona_dirs[0]
    # Replay path writes responses.jsonl
    assert (persona_dir / "responses.jsonl").exists()
    assert (persona_dir / "scored.jsonl").exists()
    assert (persona_dir / "suite_result.json").exists()


def test_run_evals_inspect_task_suite(tmp_path, monkeypatch):
    prompts = Dataset.from_list([{"question": "Q1"}])

    monkeypatch.setattr(
        "scripts.evals.run.resolve_inspect_task_ref",
        lambda task: "inspect_evals/mmlu",
    )
    monkeypatch.setattr(
        "scripts.evals.run.normalize_inspect_model_ref",
        lambda model_cfg: "hf/dummy/model",
    )
    monkeypatch.setattr(
        "scripts.evals.run.run_inspect_eval",
        lambda tasks, model_ref, eval_kwargs, log_dir: [
            {
                "eval": {"task": "inspect_evals/mmlu"},
                "results": {
                    "scores": [
                        {
                            "name": "accuracy",
                            "scorer": "match",
                            "reducer": None,
                            "metrics": {"mean": {"value": 0.75}},
                        }
                    ]
                },
                "samples": [{"id": "s1"}],
            }
        ],
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
    assert any(
        key.startswith("inspect.mmlu.") and key.endswith(".mean")
        for key in model_row
    )


def test_run_evals_inspect_task_suite_lora_auto_merge(tmp_path, monkeypatch):
    prompts = Dataset.from_list([{"question": "Q1"}])
    merged_dir = tmp_path / "merged-cache" / "merged-model"
    captured: dict[str, str] = {}

    monkeypatch.setattr(
        "scripts.evals.run.resolve_inspect_task_ref",
        lambda task: "inspect_evals/mmlu",
    )
    monkeypatch.setattr(
        "scripts.evals.run.ensure_merged_lora_model",
        lambda model_cfg, cache_dir, force_remerge, logger: merged_dir,
    )

    def _fake_inspect_eval(tasks, model_ref, eval_kwargs, log_dir):
        captured["model_ref"] = model_ref
        return [
            {
                "eval": {"task": "inspect_evals/mmlu"},
                "results": {
                    "scores": [
                        {
                            "name": "accuracy",
                            "scorer": "match",
                            "reducer": None,
                            "metrics": {"mean": {"value": 0.66}},
                        }
                    ]
                },
                "samples": [{"id": "s1"}],
            }
        ]

    monkeypatch.setattr("scripts.evals.run.run_inspect_eval", _fake_inspect_eval)

    config = EvalsConfig(
        models=[
            EvalModelConfig(
                kind="lora",
                model="dummy/model",
                adapter_path="/tmp/adapter",
            )
        ],
        suites=[InspectTaskSuiteConfig(task="mmlu")],
        output_dir=tmp_path / "evals",
        merged_model_cache_dir=tmp_path / "merged-cache",
    )

    out_dataset, result = run_evals(config, dataset=prompts)

    assert result.num_models == 1
    assert result.num_suites == 1
    assert len(out_dataset) == 1
    assert captured["model_ref"] == f"hf/{merged_dir}"
