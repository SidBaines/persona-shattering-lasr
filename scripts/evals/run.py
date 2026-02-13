"""Core runner for end-to-end model evals."""

from __future__ import annotations

import importlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from datasets import Dataset

from scripts.data_loading import format_for_inference, load_dataset_from_config
from scripts.evals.config import (
    EvalModelConfig,
    EvalsConfig,
    EvalsResult,
    InspectTaskSuiteConfig,
    ModelEvalResult,
    PersonaMetricsSuiteConfig,
    SuiteEvalResult,
)
from scripts.inference import InferenceConfig, LocalProviderConfig, run_inference
from scripts.persona_metrics import PersonaMetricsConfig, run_persona_metrics
from scripts.utils import setup_logging, write_jsonl


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _flatten_numeric_values(prefix: str, payload: Any) -> dict[str, float]:
    """Flatten nested payload into numeric key/value pairs."""
    flat: dict[str, float] = {}

    def _walk(node_prefix: str, node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                key_prefix = f"{node_prefix}.{key}" if node_prefix else str(key)
                _walk(key_prefix, value)
            return
        if isinstance(node, list):
            for i, value in enumerate(node):
                key_prefix = f"{node_prefix}.{i}" if node_prefix else str(i)
                _walk(key_prefix, value)
            return
        if isinstance(node, bool):
            return
        if isinstance(node, (int, float)):
            flat[node_prefix] = float(node)

    _walk(prefix, payload)
    return flat


def _safe_model_id(model_cfg: EvalModelConfig, idx: int) -> str:
    if model_cfg.id:
        return model_cfg.id
    base = model_cfg.model.replace("/", "_").replace(":", "_")
    if model_cfg.kind == "lora":
        adapter = (model_cfg.adapter_path or "adapter").replace("/", "_").replace(":", "_")
        return f"lora-{idx}-{base}-{adapter}"
    return f"base-{idx}-{base}"


def _inspect_model_ref(model_cfg: EvalModelConfig) -> str:
    if model_cfg.kind == "lora":
        return f"{model_cfg.model}::{model_cfg.adapter_path}"
    return model_cfg.model


def _load_question_dataset(config: EvalsConfig, dataset: Dataset | None) -> Dataset:
    if dataset is None:
        dataset = load_dataset_from_config(config.dataset)
    return format_for_inference(dataset, question_column=config.question_column)


def _generate_with_lora(
    model_cfg: EvalModelConfig,
    dataset: Dataset,
    evals_config: EvalsConfig,
) -> Dataset:
    """Generate responses using base model + LoRA adapter."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, model_cfg.dtype, None)
    if dtype is None:
        raise ValueError(f"Unsupported dtype: {model_cfg.dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.model,
        revision=model_cfg.revision,
        torch_dtype=dtype,
        device_map=model_cfg.device_map,
    )
    model = PeftModel.from_pretrained(model, model_cfg.adapter_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.model,
        revision=model_cfg.revision,
        use_fast=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    eos_ids: list[int] = []
    model_eos = getattr(model.generation_config, "eos_token_id", None)
    if isinstance(model_eos, int):
        eos_ids.append(model_eos)
    elif isinstance(model_eos, list):
        eos_ids.extend(int(token_id) for token_id in model_eos)
    if tokenizer.eos_token_id is not None:
        eos_ids.append(int(tokenizer.eos_token_id))
    eos_ids = list(dict.fromkeys(eos_ids))
    eos_token_id: int | list[int] | None
    if not eos_ids:
        eos_token_id = None
    elif len(eos_ids) == 1:
        eos_token_id = eos_ids[0]
    else:
        eos_token_id = eos_ids

    generation = evals_config.generation
    rows: list[dict[str, Any]] = []
    device = next(model.parameters()).device

    with torch.no_grad():
        for row in dataset:
            question = row["question"]
            inputs = tokenizer(
                question,
                return_tensors="pt",
                truncation=True,
            ).to(device)
            generated = model.generate(
                **inputs,
                max_new_tokens=generation.max_new_tokens,
                do_sample=generation.do_sample,
                temperature=generation.temperature,
                top_p=generation.top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_token_id,
            )
            input_len = inputs["input_ids"].shape[1]
            response = tokenizer.decode(generated[0][input_len:], skip_special_tokens=True)
            rows.append(
                {
                    "question": question,
                    "response": response,
                    "response_index": 0,
                }
            )

    return Dataset.from_list(rows)


def _generate_responses_for_model(
    model_cfg: EvalModelConfig,
    dataset: Dataset,
    evals_config: EvalsConfig,
) -> Dataset:
    if model_cfg.kind == "base":
        infer_cfg = InferenceConfig(
            model=model_cfg.model,
            provider="local",
            generation=evals_config.generation,
            local=LocalProviderConfig(
                dtype=model_cfg.dtype,
                device_map=model_cfg.device_map,
                revision=model_cfg.revision,
                prompt_format="auto",
            ),
        )
        result_dataset, _ = run_inference(infer_cfg, dataset=dataset)
        return result_dataset

    return _generate_with_lora(model_cfg, dataset, evals_config)


def _resolve_inspect_task_name(task: str, task_name: str | None) -> str:
    if task_name:
        return task_name
    if task == "mmlu":
        return "mmlu"
    return task.split(":")[0].replace("/", "_")


def _call_hook(task: str, model_ref: str, task_params: dict[str, Any]) -> dict[str, Any]:
    if ":" not in task:
        raise ValueError(
            "Task hook must be in 'module.path:function' format when not using "
            "the built-in alias 'mmlu'."
        )
    module_path, func_name = task.split(":", 1)
    module = importlib.import_module(module_path)
    hook = getattr(module, func_name, None)
    if not callable(hook):
        raise ValueError(f"Inspect task hook '{task}' is not callable.")
    return hook(model_ref=model_ref, task_params=task_params)


def _run_inspect_eval(
    task: str,
    model_ref: str,
    task_params: dict[str, Any],
) -> dict[str, Any]:
    """Run an Inspect task via inspect_ai API or a Python hook."""
    if ":" in task and task != "mmlu":
        return _call_hook(task, model_ref, task_params)

    inspect_module = importlib.import_module("inspect_ai")
    eval_fn = getattr(inspect_module, "eval", None)
    if eval_fn is None:
        eval_module = importlib.import_module("inspect_ai._eval.eval")
        eval_fn = getattr(eval_module, "eval")

    # Built-in alias mapping
    task_ref = "mmlu" if task == "mmlu" else task

    attempts: list[dict[str, Any]] = [
        {"task": task_ref, "model": model_ref, **task_params},
        {"tasks": [task_ref], "model": model_ref, **task_params},
    ]
    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            result = eval_fn(**kwargs)
            if isinstance(result, dict):
                return result
            if hasattr(result, "model_dump"):
                return result.model_dump()
            if isinstance(result, list):
                payload: list[Any] = []
                for item in result:
                    if hasattr(item, "model_dump"):
                        payload.append(item.model_dump())
                    else:
                        payload.append(item)
                return {"results": payload}
            return {"result": str(result)}
        except Exception as exc:  # pragma: no cover - defensive for API drift
            last_error = exc

    if last_error is not None:
        raise last_error
    raise RuntimeError("Inspect eval call failed with unknown error.")


def run_evals(
    config: EvalsConfig,
    dataset: Dataset | None = None,
) -> tuple[Dataset, EvalsResult]:
    """Run end-to-end eval suites for one or more model targets."""
    logger = setup_logging()
    started_at = datetime.now()

    output_dir = config.output_dir or Path("scratch") / (
        f"evals-{started_at.strftime('%Y%m%d-%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    question_dataset = _load_question_dataset(config, dataset)
    model_results: list[ModelEvalResult] = []
    combined_records: list[dict[str, Any]] = []
    leaderboard_by_model: dict[str, dict[str, Any]] = {}

    for model_index, model_cfg in enumerate(config.models):
        model_id = _safe_model_id(model_cfg, model_index)
        logger.info("Running evals for model %s (%s)", model_id, model_cfg.kind)
        model_output_dir = output_dir / model_id
        model_output_dir.mkdir(parents=True, exist_ok=True)

        responses_dataset: Dataset | None = None
        model_result = ModelEvalResult(
            model_id=model_id,
            kind=model_cfg.kind,
            model=model_cfg.model,
            adapter_path=model_cfg.adapter_path,
        )

        leaderboard_by_model[model_id] = {"model_id": model_id}

        for suite in config.suites:
            if isinstance(suite, PersonaMetricsSuiteConfig):
                suite_name = "persona_metrics"
                suite_dir = model_output_dir / suite_name
                suite_dir.mkdir(parents=True, exist_ok=True)
                suite_result = SuiteEvalResult(
                    suite_type=suite.type,
                    suite_name=suite_name,
                    model_id=model_id,
                )

                try:
                    if responses_dataset is None:
                        responses_dataset = _generate_responses_for_model(
                            model_cfg=model_cfg,
                            dataset=question_dataset,
                            evals_config=config,
                        )

                    responses_path = write_jsonl(
                        responses_dataset.to_list(),
                        suite_dir / "responses.jsonl",
                    )

                    metrics_config = PersonaMetricsConfig(
                        evaluations=suite.evaluations,
                        response_column=suite.response_column,
                        question_column=suite.question_column,
                        metrics_key=suite.metrics_key,
                        judge=suite.judge,
                    )
                    scored_dataset, metrics_result = run_persona_metrics(
                        metrics_config,
                        dataset=responses_dataset,
                    )

                    scored_path = write_jsonl(
                        scored_dataset.to_list(),
                        suite_dir / "scored.jsonl",
                    )
                    suite_payload = {
                        "suite_type": suite.type,
                        "suite_name": suite_name,
                        "model_id": model_id,
                        "num_samples": metrics_result.num_samples,
                        "aggregates": metrics_result.aggregates,
                    }
                    suite_result_path = _write_json(
                        suite_dir / "suite_result.json",
                        suite_payload,
                    )

                    suite_result.num_samples = metrics_result.num_samples
                    suite_result.aggregates = dict(metrics_result.aggregates)
                    suite_result.artifacts = {
                        "responses": str(responses_path),
                        "scored": str(scored_path),
                        "suite_result": str(suite_result_path),
                    }

                    prefixed = _flatten_numeric_values(
                        "persona_metrics", metrics_result.aggregates
                    )
                    leaderboard_by_model[model_id].update(prefixed)

                    for row in scored_dataset:
                        out = dict(row)
                        out["model_id"] = model_id
                        out["suite_type"] = suite.type
                        combined_records.append(out)
                except Exception as exc:
                    suite_result.error = str(exc)
                    if not config.continue_on_error:
                        raise

                model_result.suites.append(suite_result)
                continue

            if isinstance(suite, InspectTaskSuiteConfig):
                task_name = _resolve_inspect_task_name(suite.task, suite.task_name)
                suite_name = f"inspect.{task_name}"
                suite_dir = model_output_dir / suite_name
                suite_dir.mkdir(parents=True, exist_ok=True)
                suite_result = SuiteEvalResult(
                    suite_type=suite.type,
                    suite_name=suite_name,
                    model_id=model_id,
                )

                try:
                    inspect_payload = _run_inspect_eval(
                        task=suite.task,
                        model_ref=_inspect_model_ref(model_cfg),
                        task_params=dict(suite.task_params),
                    )
                    aggregates = _flatten_numeric_values(
                        f"inspect.{task_name}",
                        inspect_payload,
                    )
                    suite_json_path = _write_json(
                        suite_dir / "suite_result.json",
                        {
                            "suite_type": suite.type,
                            "suite_name": suite_name,
                            "model_id": model_id,
                            "task": suite.task,
                            "task_params": suite.task_params,
                            "payload": inspect_payload,
                        },
                    )

                    suite_result.aggregates = aggregates
                    suite_result.artifacts = {"suite_result": str(suite_json_path)}
                    suite_result.metadata = {
                        "task": suite.task,
                        "task_params": suite.task_params,
                    }

                    leaderboard_by_model[model_id].update(aggregates)
                    combined_records.append(
                        {
                            "model_id": model_id,
                            "suite_type": suite.type,
                            "suite_name": suite_name,
                            "inspect_payload": inspect_payload,
                        }
                    )
                except Exception as exc:
                    suite_result.error = str(exc)
                    if not config.continue_on_error:
                        raise

                model_result.suites.append(suite_result)
                continue

        model_results.append(model_result)

    leaderboard = [leaderboard_by_model[key] for key in sorted(leaderboard_by_model)]
    leaderboard_path = _write_json(output_dir / "leaderboard.json", {"rows": leaderboard})
    summary_payload = {
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now().isoformat(),
        "num_models": len(config.models),
        "num_suites": len(config.suites),
        "num_rows": len(combined_records),
        "leaderboard_path": str(leaderboard_path),
        "models": [model.model_dump() for model in config.models],
        "suites": [suite.model_dump() for suite in config.suites],
    }
    summary_path = _write_json(output_dir / "summary.json", summary_payload)

    result = EvalsResult(
        output_dir=output_dir,
        num_models=len(config.models),
        num_suites=len(config.suites),
        num_rows=len(combined_records),
        model_results=model_results,
        leaderboard=leaderboard,
        summary_path=summary_path,
    )

    return Dataset.from_list(combined_records), result
