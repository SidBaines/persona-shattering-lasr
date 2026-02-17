"""CLI entry point for end-to-end eval runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.common.config import DatasetConfig, GenerationConfig
from scripts.evals.config import (
    EvalModelConfig,
    EvalsConfig,
    InspectTaskSuiteConfig,
    PersonaMetricsSuiteConfig,
)
from scripts.evals.run import run_evals
from scripts.persona_metrics import JudgeLLMConfig


def _parse_lora_spec(value: str) -> EvalModelConfig:
    parts = value.split("::")
    if len(parts) == 2:
        base_model, adapter_path = parts
        model_id = None
    elif len(parts) == 3:
        model_id, base_model, adapter_path = parts
    else:
        raise ValueError(
            "Invalid --lora-model format. Use BASE_MODEL::ADAPTER_PATH or "
            "MODEL_ID::BASE_MODEL::ADAPTER_PATH."
        )
    return EvalModelConfig(
        id=model_id,
        kind="lora",
        model=base_model,
        adapter_path=adapter_path,
    )


def _parse_inspect_task_spec(value: str) -> InspectTaskSuiteConfig:
    # Format: task_ref or task_ref::{"max_samples": 100}
    if "::" not in value:
        return InspectTaskSuiteConfig(task=value)
    task_ref, params_json = value.split("::", 1)
    eval_kwargs = json.loads(params_json)
    if not isinstance(eval_kwargs, dict):
        raise ValueError("--inspect-task JSON must decode to an object.")
    return InspectTaskSuiteConfig(task=task_ref, eval_kwargs=eval_kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end evals on base and/or LoRA models.",
    )

    # Model targets
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Base model name/path. Repeat flag for multiple models.",
    )
    parser.add_argument(
        "--lora-model",
        action="append",
        default=[],
        help=(
            "LoRA model spec. Format: BASE_MODEL::ADAPTER_PATH or "
            "MODEL_ID::BASE_MODEL::ADAPTER_PATH. Repeat flag for multiple models."
        ),
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Torch dtype for local loading (default: bfloat16).",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device map for local loading (default: auto).",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Model revision for base model loading (default: main).",
    )

    # Dataset
    parser.add_argument(
        "--dataset-source",
        type=str,
        choices=["huggingface", "local"],
        default="huggingface",
        help="Dataset source type (default: huggingface).",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Dataset name for source=huggingface.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Dataset path for source=local.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional sample cap.",
    )
    parser.add_argument(
        "--question-column",
        type=str,
        default="question",
        help="Question column in the source dataset (default: question).",
    )

    # Generation (deterministic defaults)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max new tokens for generation (default: 256).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature (default: 0.0).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Generation top-p (default: 1.0).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Generation batch size (default: 8).",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling (default: disabled for deterministic comparison).",
    )

    # Suite selection
    parser.add_argument(
        "--persona-evaluations",
        nargs="+",
        default=None,
        help=(
            "Persona metrics to run. If provided, enables the persona_metrics suite "
            "(e.g., count_o coherence)."
        ),
    )
    parser.add_argument(
        "--persona-judge-provider",
        type=str,
        choices=["openai", "openrouter", "anthropic"],
        default="openai",
        help="Judge provider for persona metric LLM-as-judge metrics.",
    )
    parser.add_argument(
        "--persona-judge-model",
        type=str,
        default="gpt-4o-mini",
        help="Judge model for persona metric LLM-as-judge metrics.",
    )
    parser.add_argument(
        "--persona-judge-max-concurrent",
        type=int,
        default=10,
        help="Max concurrent judge requests for persona metric suite.",
    )
    parser.add_argument(
        "--inspect-task",
        action="append",
        default=[],
        help=(
            "Inspect task spec. Format: TASK_REF or TASK_REF::JSON_EVAL_KWARGS. "
            "Repeat for multiple tasks. Example: "
            "--inspect-task inspect_evals/mmlu_0_shot::'{\"max_samples\": 100}'. "
            "Alias: mmlu -> inspect_evals/mmlu_0_shot (requires inspect_evals package)."
        ),
    )

    # Output and behavior
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: scratch/evals-<timestamp>).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running remaining suites/models if one suite fails.",
    )
    parser.add_argument(
        "--merged-model-cache-dir",
        type=str,
        default="scratch/merged_lora_models",
        help=(
            "Directory for cached merged LoRA models used by inspect_task suites "
            "(default: scratch/merged_lora_models)."
        ),
    )
    parser.add_argument(
        "--force-remerge-lora",
        action="store_true",
        help="Force rebuilding merged LoRA cache entries for inspect_task suites.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    models: list[EvalModelConfig] = []
    for model_name in args.model:
        models.append(
            EvalModelConfig(
                kind="base",
                model=model_name,
                dtype=args.dtype,
                device_map=args.device_map,
                revision=args.revision,
            )
        )
    for spec in args.lora_model:
        model_cfg = _parse_lora_spec(spec)
        model_cfg.dtype = args.dtype
        model_cfg.device_map = args.device_map
        model_cfg.revision = args.revision
        models.append(model_cfg)

    if not models:
        raise ValueError("Specify at least one --model and/or --lora-model.")

    suites = []
    if args.persona_evaluations is not None:
        suites.append(
            PersonaMetricsSuiteConfig(
                evaluations=list(args.persona_evaluations),
                judge=JudgeLLMConfig(
                    provider=args.persona_judge_provider,
                    model=args.persona_judge_model,
                    max_concurrent=args.persona_judge_max_concurrent,
                ),
            )
        )
    for task_spec in args.inspect_task:
        suites.append(_parse_inspect_task_spec(task_spec))

    if not suites:
        raise ValueError(
            "No eval suites requested. Provide --persona-evaluations and/or --inspect-task."
        )

    config = EvalsConfig(
        models=models,
        suites=suites,
        dataset=DatasetConfig(
            source=args.dataset_source,
            name=args.dataset_name,
            path=args.dataset_path,
            max_samples=args.max_samples,
        ),
        question_column=args.question_column,
        generation=GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
            batch_size=args.batch_size,
            num_responses_per_prompt=1,
        ),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        merged_model_cache_dir=Path(args.merged_model_cache_dir),
        force_remerge_lora=args.force_remerge_lora,
        continue_on_error=args.continue_on_error,
    )

    result_dataset, result = run_evals(config)
    print(f"\nEvals complete.")
    print(f"Models: {result.num_models}")
    print(f"Suites: {result.num_suites}")
    print(f"Rows:   {result.num_rows}")
    if result.output_dir:
        print(f"Output: {result.output_dir}")
    if result.summary_path:
        print(f"Summary: {result.summary_path}")
    if result.leaderboard:
        print("\nLeaderboard:")
        for row in result.leaderboard:
            print(f"  - {row.get('model_id')}: {len(row.keys()) - 1} metric(s)")
    # Keep variable used to make intent clear for callers reading output.
    _ = result_dataset


if __name__ == "__main__":
    main()
