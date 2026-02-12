#!/usr/bin/env python3
"""Unified persona training pipeline.

Reads a YAML config file and orchestrates all four pipeline stages:
  1. Inference  — local HF model generates responses for a dataset
  2. Editing    — API LLM rewrites responses in the persona's style
  3. Evaluation — style metrics computed on original + edited responses
  4. Training   — LoRA fine-tuning on the evaluated dataset

Each stage can be individually enabled or disabled in the YAML's `stages:`
block. Disabled stages load their expected output from scratch/<run_id>/
rather than running.

Usage:
    uv run python scripts/experiments/persona_pipelines/persona_training.py
    uv run python scripts/experiments/persona_pipelines/persona_training.py \\
        --config scripts/experiments/persona_pipelines/configs/sf_guy.yaml
    uv run python scripts/experiments/persona_pipelines/persona_training.py \\
        --config configs/sf_guy.yaml --run-id sf-guy-20250101-120000
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import yaml
from datasets import Dataset
from dotenv import load_dotenv

from scripts.common.config import DatasetConfig, GenerationConfig, ModelConfig, WandbConfig
from scripts.editing import EditingConfig, OpenAIProviderConfig, run_editing
from scripts.editing.config import RetryConfig as EditingRetryConfig
from scripts.evaluation import EvaluationConfig, JudgeLLMConfig, run_evaluation
from scripts.inference import InferenceConfig, run_inference
from scripts.training import (
    LoraConfig,
    SftConfig,
    TrainingConfig,
    TrainingEvaluationConfig,
    run_training,
)
from scripts.utils import read_jsonl, write_jsonl

# ── Fixed output file names under scratch/<run_id>/ ─────────────────────────

INFERENCE_OUTPUT = "inference_output.jsonl"
INFERENCE_PAIRS_OUTPUT = "question_response_pairs.jsonl"
EDITING_OUTPUT = "edited_dataset.jsonl"
EVALUATION_OUTPUT = "edited_evaluated.jsonl"
CHECKPOINTS_SUBDIR = "checkpoints"


# ── Path helpers ─────────────────────────────────────────────────────────────


def _scratch_dir(run_id: str) -> Path:
    return Path("scratch") / run_id


def _inference_path(run_id: str) -> Path:
    return _scratch_dir(run_id) / INFERENCE_OUTPUT


def _editing_path(run_id: str) -> Path:
    return _scratch_dir(run_id) / EDITING_OUTPUT


def _evaluation_path(run_id: str) -> Path:
    return _scratch_dir(run_id) / EVALUATION_OUTPUT


def _checkpoint_dir(run_id: str) -> Path:
    return _scratch_dir(run_id) / CHECKPOINTS_SUBDIR


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_config = script_dir / "configs" / "sf_guy.yaml"
    parser = argparse.ArgumentParser(
        description="Run the unified persona training pipeline from a YAML config."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(default_config),
        help=f"Path to YAML config file (default: {default_config}).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Override the run ID from config (auto-generates if neither provides one).",
    )
    return parser.parse_args()


# ── Config loading ────────────────────────────────────────────────────────────


def _load_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_run_id(cfg: dict[str, Any], cli_run_id: str | None) -> str:
    if cli_run_id:
        return cli_run_id
    yaml_run_id = cfg.get("run_id")
    if yaml_run_id:
        return str(yaml_run_id)
    persona_name = cfg.get("persona", {}).get("name", "persona")
    return f"{persona_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


# ── Stage-skipping data loader ────────────────────────────────────────────────


def _load_dataset_from_path(path: Path, stage_name: str) -> Dataset:
    """Load a JSONL file into an HF Dataset; raises clearly if it's missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Stage '{stage_name}' is disabled but its expected input was not found.\n"
            f"  Expected: {path}\n"
            f"  Either enable the preceding stage or ensure this file exists."
        )
    return Dataset.from_list(read_jsonl(path))


# ── Per-stage config builders ─────────────────────────────────────────────────


def _build_inference_config(cfg: dict[str, Any], run_id: str) -> InferenceConfig:
    inf = cfg.get("inference", {})
    gen = inf.get("generation", {})
    ds = cfg.get("dataset", {})
    return InferenceConfig(
        model=cfg["base_model"],
        provider=inf.get("provider", "local"),
        dataset=DatasetConfig(
            source=ds.get("source", "huggingface"),
            name=ds.get("name"),
            split=ds.get("split", "train"),
            max_samples=ds.get("max_samples"),
        ),
        generation=GenerationConfig(
            max_new_tokens=gen.get("max_new_tokens", 1024),
            temperature=gen.get("temperature", 0.7),
            top_p=gen.get("top_p", 0.9),
            do_sample=gen.get("do_sample", True),
            batch_size=gen.get("batch_size", 16),
            num_responses_per_prompt=gen.get("num_responses_per_prompt", 1),
        ),
        output_path=_inference_path(run_id),
    )


def _build_editing_config(cfg: dict[str, Any], run_id: str) -> EditingConfig:
    ed = cfg.get("editing", {})
    retry = ed.get("retry", {})
    prompt_template = cfg.get("persona", {}).get(
        "editing_prompt_template", "sf_guy_casual_grammar"
    )
    return EditingConfig(
        provider=ed.get("provider", "openai"),
        model=ed.get("model", "gpt-5-nano-2025-08-07"),
        prompt_template=prompt_template,
        openai=OpenAIProviderConfig(
            max_tokens=ed.get("max_tokens", 50000),
        ),
        max_concurrent=ed.get("max_concurrent", 20),
        retry=EditingRetryConfig(
            max_retries=retry.get("max_retries", 8),
            backoff_factor=retry.get("backoff_factor", 1.5),
        ),
        output_path=_editing_path(run_id),
    )


def _build_training_config(cfg: dict[str, Any], run_id: str) -> TrainingConfig:
    tr = cfg.get("training", {})
    lora_raw = tr.get("lora", {})
    sft_raw = tr.get("sft", {})
    wandb_raw = tr.get("wandb", {})
    eval_raw = tr.get("evaluation", {})
    judge_raw = eval_raw.get("judge", {})
    return TrainingConfig(
        model=ModelConfig(
            name=cfg["base_model"],
            dtype="bfloat16",
            device_map="auto",
        ),
        lora=LoraConfig(
            r=lora_raw.get("r", 16),
            lora_alpha=lora_raw.get("lora_alpha", 16),
            lora_dropout=lora_raw.get("lora_dropout", 0.0),
        ),
        sft=SftConfig(
            num_train_epochs=sft_raw.get("num_train_epochs", 10),
            per_device_train_batch_size=sft_raw.get("per_device_train_batch_size", 4),
            gradient_accumulation_steps=sft_raw.get("gradient_accumulation_steps", 4),
            learning_rate=sft_raw.get("learning_rate", 1e-4),
            lr_scheduler_type=sft_raw.get("lr_scheduler_type", "cosine"),
            warmup_ratio=sft_raw.get("warmup_ratio", 0.05),
            max_seq_length=sft_raw.get("max_seq_length", 1024),
            fp16=sft_raw.get("fp16", False),
            bf16=sft_raw.get("bf16", True),
        ),
        wandb=WandbConfig(
            enabled=wandb_raw.get("enabled", True),
            project=wandb_raw.get("project", "persona-shattering-v1"),
            entity=wandb_raw.get("entity"),
            tags=wandb_raw.get("tags", []),
            group=wandb_raw.get("group"),
        ),
        evaluation=TrainingEvaluationConfig(
            enabled=eval_raw.get("enabled", True),
            evaluations=eval_raw.get("metrics", ["lowercase_density", "punctuation_density"]),
            judge=JudgeLLMConfig(
                provider=judge_raw.get("provider", "openai"),
                model=judge_raw.get("model", "gpt-5-nano-2025-08-07"),
            ),
            num_samples=eval_raw.get("num_samples", 20),
            max_new_tokens=eval_raw.get("max_new_tokens", 128),
            eval_every_n_epochs=eval_raw.get("eval_every_n_epochs", 1),
        ),
        checkpoint_dir=_checkpoint_dir(run_id),
        val_split=tr.get("val_split", 0.1),
        seed=tr.get("seed", 42),
    )


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = _parse_args()

    # HuggingFace cache directories (matches existing pipeline scripts)
    for cache_subdir in ("", "hub", "datasets", "transformers"):
        Path("/workspace/.cache/huggingface", cache_subdir).mkdir(
            parents=True, exist_ok=True
        )

    load_dotenv()

    cfg = _load_config(args.config)
    run_id = _resolve_run_id(cfg, args.run_id)
    stages = cfg.get("stages", {})

    scratch = _scratch_dir(run_id)
    scratch.mkdir(parents=True, exist_ok=True)

    persona_name = cfg.get("persona", {}).get("name", "unknown")
    enabled = [name for name, on in stages.items() if on]
    skipped = [name for name, on in stages.items() if not on]

    print(f"\n{'=' * 60}")
    print("PERSONA TRAINING PIPELINE")
    print(f"Persona : {persona_name}")
    print(f"Run ID  : {run_id}")
    print(f"Config  : {args.config}")
    print(f"Output  : {scratch}")
    print(f"Enabled : {', '.join(enabled) or 'none'}")
    print(f"Skipped : {', '.join(skipped) or 'none'}")
    print(f"{'=' * 60}\n")

    # ── Stage 1: Inference ────────────────────────────────────────────────────

    if stages.get("inference", True):
        print("[Stage 1/4] Running inference...")
        inference_config = _build_inference_config(cfg, run_id)
        inference_dataset, inference_result = run_inference(inference_config)

        pairs_path = scratch / INFERENCE_PAIRS_OUTPUT
        write_jsonl(
            [
                {"question": rec["question"], "response": rec["response"]}
                for rec in inference_dataset.to_list()
            ],
            pairs_path,
        )
        print(f"  Generated {inference_result.num_samples} responses")
        print(f"  Saved to : {inference_result.output_path}")
        print(f"  Pairs    : {pairs_path}")
    else:
        print("[Stage 1/4] Inference SKIPPED — loading from disk...")
        inference_dataset = _load_dataset_from_path(_inference_path(run_id), "inference")
        print(f"  Loaded {len(inference_dataset)} records from {_inference_path(run_id)}")

    # ── Stage 2: Editing ──────────────────────────────────────────────────────

    if stages.get("editing", True):
        print("\n[Stage 2/4] Running editing...")
        editing_config = _build_editing_config(cfg, run_id)
        edited_dataset, editing_result = run_editing(editing_config, dataset=inference_dataset)
        print(
            f"  Edited {editing_result.num_samples} responses "
            f"({editing_result.num_failed} failed)"
        )
        print(f"  Saved to : {editing_result.output_path}")
    else:
        print("\n[Stage 2/4] Editing SKIPPED — loading from disk...")
        edited_dataset = _load_dataset_from_path(_editing_path(run_id), "editing")
        print(f"  Loaded {len(edited_dataset)} records from {_editing_path(run_id)}")

    # ── Stage 3: Evaluation ───────────────────────────────────────────────────

    if stages.get("evaluation", True):
        print("\n[Stage 3/4] Running evaluation...")
        ev = cfg.get("evaluation", {})
        metrics = ev.get("metrics", ["lowercase_density", "punctuation_density"])

        response_eval_config = EvaluationConfig(
            evaluations=metrics,
            response_column="response",
            question_column="question",
            metrics_key="response_style_metrics",
        )
        edited_eval_config = EvaluationConfig(
            evaluations=metrics,
            response_column="edited_response",
            question_column="question",
            metrics_key="edited_style_metrics",
        )

        response_eval_dataset, response_eval_result = run_evaluation(
            response_eval_config, dataset=edited_dataset
        )
        evaluated_dataset, edited_eval_result = run_evaluation(
            edited_eval_config, dataset=response_eval_dataset
        )

        # Compute per-row deltas (edited - original) for numeric metrics
        records_with_metrics = evaluated_dataset.to_list()
        for record in records_with_metrics:
            response_metrics = record.get("response_style_metrics", {})
            edited_metrics = record.get("edited_style_metrics", {})
            delta_metrics: dict[str, float | int] = {}
            for key in sorted(set(response_metrics).intersection(edited_metrics)):
                original_value = response_metrics[key]
                edited_value = edited_metrics[key]
                if isinstance(original_value, (int, float)) and isinstance(
                    edited_value, (int, float)
                ):
                    delta_metrics[f"{key}.delta"] = edited_value - original_value
            record["style_metrics_delta"] = delta_metrics

        eval_output_path = _evaluation_path(run_id)
        write_jsonl(records_with_metrics, eval_output_path)
        evaluated_dataset = Dataset.from_list(records_with_metrics)

        print(f"  Evaluated {response_eval_result.num_samples} rows on 'response'")
        print(f"  Evaluated {edited_eval_result.num_samples} rows on 'edited_response'")
        print(f"  Added per-row deltas in 'style_metrics_delta'")
        print(f"  Saved to : {eval_output_path}")
    else:
        print("\n[Stage 3/4] Evaluation SKIPPED — loading from disk...")
        evaluated_dataset = _load_dataset_from_path(_evaluation_path(run_id), "evaluation")
        print(f"  Loaded {len(evaluated_dataset)} records from {_evaluation_path(run_id)}")

    # ── Stage 4: Training ─────────────────────────────────────────────────────

    if stages.get("training", True):
        print("\n[Stage 4/4] Running training...")
        training_config = _build_training_config(cfg, run_id)
        _, training_result = run_training(training_config, dataset=evaluated_dataset)
        print(f"  Trained on {training_result.num_train_samples} samples")
        print(f"  Validation : {training_result.num_val_samples} samples")
        print(f"  Model saved: {training_result.checkpoint_path}")
    else:
        print("\n[Stage 4/4] Training SKIPPED.")

    # ── Summary ───────────────────────────────────────────────────────────────

    print(f"\n{'=' * 60}")
    print("PIPELINE COMPLETE")
    print(f"Persona : {persona_name}")
    print(f"Run ID  : {run_id}")
    print(f"Output  : {scratch}")
    if stages.get("training", True):
        print(f"Model   : {_checkpoint_dir(run_id)}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
