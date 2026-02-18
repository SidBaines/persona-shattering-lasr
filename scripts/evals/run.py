"""Core runner: thin wrapper around ``lm_eval.evaluator.simple_evaluate``."""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

import lm_eval.evaluator
from lm_eval.tasks import TaskManager

from scripts.evals.config import EvalConfig
from scripts.evals.lora_merge import merge_adapters
from scripts.evals.tasks import TASKS_DIR

logger = logging.getLogger(__name__)


def _build_model_args(
    config: EvalConfig,
    *,
    peft: str | None = None,
    pretrained_override: str | None = None,
) -> str:
    """Build the ``model_args`` string expected by lm_eval's HF backend.

    Parameters
    ----------
    config:
        Eval configuration.
    peft:
        If set, added as ``peft=<path>`` for native single-adapter support.
    pretrained_override:
        If set, replaces ``config.model`` as the ``pretrained`` value
        (used when pointing at a temp merged model directory).
    """
    parts: dict[str, str] = {}
    parts["pretrained"] = pretrained_override or config.model

    if peft:
        parts["peft"] = peft

    # Merge in any extra model_args from config
    for k, v in config.model_args.items():
        if k not in parts:
            parts[k] = v

    return ",".join(f"{k}={v}" for k, v in parts.items())


def _save_results(results: dict[str, Any], output_path: Path) -> None:
    """Write lm_eval results to disk in native JSON format."""
    output_path.mkdir(parents=True, exist_ok=True)
    results_file = output_path / "results.json"

    # Extract the serializable portion
    output = {}
    for key in ("results", "configs", "versions", "n-shot", "higher_is_better"):
        if key in results:
            output[key] = results[key]

    results_file.write_text(
        json.dumps(output, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("Results saved to %s", results_file)


def run_eval(config: EvalConfig) -> dict[str, Any]:
    """Run an evaluation using lm-evaluation-harness.

    Handles three adapter scenarios:
    - No adapters: plain base model
    - Single adapter at scale=1.0: native ``peft=`` in model_args
    - Scaled / multi-adapter: merge to temp dir, cleanup after

    Parameters
    ----------
    config:
        Evaluation configuration.

    Returns
    -------
    dict
        The full results dict from ``lm_eval.evaluator.simple_evaluate``.
    """
    task_manager = TaskManager(include_path=str(TASKS_DIR))
    temp_dir: str | None = None

    try:
        # Resolve model args based on adapter configuration
        if not config.adapters:
            logger.info("No adapters — using base model: %s", config.model)
            model_args = _build_model_args(config)

        elif not config.needs_merge:
            # Single adapter at scale=1.0 — use native peft= support
            adapter = config.adapters[0]
            logger.info(
                "Single adapter at scale=1.0 — using native peft: %s",
                adapter.path,
            )
            model_args = _build_model_args(config, peft=adapter.path)

        else:
            # Scaled or multi-adapter — merge to temp directory
            temp_dir = tempfile.mkdtemp(prefix="lm_eval_merged_")
            logger.info(
                "Merging %d adapter(s) to temp dir: %s",
                len(config.adapters),
                temp_dir,
            )
            merge_adapters(
                base_model=config.model,
                adapters=config.adapters,
                output_dir=Path(temp_dir),
                dtype=config.model_args.get("dtype", "bfloat16"),
            )
            model_args = _build_model_args(config, pretrained_override=temp_dir)

        # Build gen_kwargs
        gen_kwargs: dict[str, Any] = {}
        if config.max_gen_toks != 256:
            gen_kwargs["max_gen_toks"] = config.max_gen_toks
        if config.temperature != 0.0:
            gen_kwargs["temperature"] = config.temperature

        logger.info("model_args: %s", model_args)
        logger.info("tasks: %s", config.tasks)

        results = lm_eval.evaluator.simple_evaluate(
            model="hf",
            model_args=model_args,
            tasks=config.tasks,
            num_fewshot=config.num_fewshot,
            batch_size=config.batch_size,
            device=config.device,
            limit=config.limit,
            log_samples=config.log_samples,
            task_manager=task_manager,
            apply_chat_template=config.apply_chat_template,
            gen_kwargs=gen_kwargs or None,
        )

        if config.output_path:
            _save_results(results, config.output_path)

        return results

    finally:
        if temp_dir:
            logger.info("Cleaning up temp merged model: %s", temp_dir)
            shutil.rmtree(temp_dir, ignore_errors=True)
