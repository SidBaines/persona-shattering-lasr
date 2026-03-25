"""Stage-separated OCT conscientiousness evaluation.

This script keeps rollout generation and judge scoring as separate cached stages:

1. Rehydrate OCT source artifacts from a HF dataset repo
2. Generate rollouts for base / DPO / SFT / persona variants and save + upload them
3. Score the saved rollouts with conscientiousness_v2 and save + upload them
4. Aggregate scores and render a bar chart
"""

from __future__ import annotations

import asyncio
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from scripts.experiments.oct_pipeline.eval_oct_conscientiousness_hf import (
    _MODEL_COLORS,
    _MODEL_VARIANTS,
    _config_hash,
    _default_base_model_ref,
    _default_constitution,
    _ensure_base_model,
    _ensure_run_config,
    _ensure_stage_available,
    _hydrate_source_artifacts,
    _prepare_prompt_subset,
    _publish_stage,
    _read_json,
    _resolve_source_run_dir,
    _load_source_run_config,
    _write_jsonl,
)
from src_dev.common.config import GenerationConfig
from src_dev.inference import InferenceConfig, LocalProviderConfig, run_inference
from src_dev.persona_metrics import (
    JudgeLLMConfig,
    PersonaMetricsConfig,
    create_persona_metrics,
)
from src_dev.utils import setup_logging, write_jsonl

matplotlib.use("Agg")
import matplotlib.pyplot as plt

load_dotenv()


_DEFAULT_OUTPUT_ROOT = Path("scratch/oct_pipeline_eval_runs")
_DEFAULT_MODEL_CACHE_ROOT = Path("/root/.cache/models")
_DEFAULT_RUN_PREFIX = "oct-conscientiousness-staged"
_DEFAULT_HF_LOG_REPO = "hf://datasets/persona-shattering-lasr/eval-logs"
_DEFAULT_PROMPT_DATASET = "mirlab/TRAIT"
_DEFAULT_PROMPT_SPLIT = "Conscientiousness"


class PromptSourceConfig(BaseModel):
    """Prompt dataset configuration."""

    source: str = "huggingface"
    name: str = _DEFAULT_PROMPT_DATASET
    split: str = _DEFAULT_PROMPT_SPLIT
    path: str | None = None
    question_column: str | None = "question"
    max_samples: int = 100
    seed: int = 223457


class ExperimentConfig(BaseModel):
    """Semantic config for the staged OCT conscientiousness eval."""

    source_hf_repo: str
    source_run_dir: str | None = None
    results_hf_repo: str | None = None
    constitution: str | None = None
    base_model_ref: str | None = None
    model_cache_root: str = str(_DEFAULT_MODEL_CACHE_ROOT)
    prompt_source: PromptSourceConfig = Field(default_factory=PromptSourceConfig)
    judge: JudgeLLMConfig = Field(
        default_factory=lambda: JudgeLLMConfig(
            provider="openai",
            model="gpt-5-nano-2025-08-07",
            temperature=0.0,
            max_tokens=4096,
            max_concurrent=10,
        )
    )
    generation: GenerationConfig = Field(
        default_factory=lambda: GenerationConfig(
            max_new_tokens=4096,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            batch_size=8,
            num_responses_per_prompt=1,
        )
    )
    hf_log_dir: str | None = _DEFAULT_HF_LOG_REPO


def _short_hash(text: str, length: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def _make_run_id(config: ExperimentConfig, source_run_dir: str, constitution: str, base_model_ref: str) -> str:
    payload = {
        **config.model_dump(),
        "source_run_dir": source_run_dir,
        "constitution": constitution,
        "base_model_ref": base_model_ref,
    }
    digest = _short_hash(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return (
        f"{_DEFAULT_RUN_PREFIX}-{source_run_dir.replace('_', '-')}-"
        f"n{config.prompt_source.max_samples}-s{config.prompt_source.seed}-{digest}"
    )


def _adapter_dir(out_path: Path, constitution: str, kind: str) -> Path:
    return out_path / "source_oct_run" / "lora" / f"{constitution}-{kind}"


def _prompt_rows(prompt_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in prompt_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text:
            rows.append(json.loads(text))
    return rows


def _rollouts_dir(out_path: Path) -> Path:
    return out_path / "rollouts"


def _variant_rollout_path(out_path: Path, model_spec: str) -> Path:
    return _rollouts_dir(out_path) / f"{model_spec}.jsonl"


def _combined_rollout_path(out_path: Path) -> Path:
    return _rollouts_dir(out_path) / "combined_rollouts.jsonl"


def _scored_rollout_path(out_path: Path) -> Path:
    return out_path / "analysis" / "scored_rollouts.jsonl"


def _summary_csv_path(out_path: Path) -> Path:
    return out_path / "analysis" / "conscientiousness_by_model_summary.csv"


def _long_csv_path(out_path: Path) -> Path:
    return out_path / "analysis" / "conscientiousness_by_model.csv"


def _figure_path(out_path: Path) -> Path:
    return out_path / "figures" / "conscientiousness_bar_chart.png"


def _rollout_stage_artifacts(out_path: Path) -> list[dict[str, Any]]:
    artifacts = [
        {"path": _combined_rollout_path(out_path), "kind": "file"},
    ]
    for spec_name, _, _ in _MODEL_VARIANTS:
        artifacts.append({"path": _variant_rollout_path(out_path, spec_name), "kind": "file"})
    return artifacts


def _scoring_stage_artifacts(out_path: Path) -> list[dict[str, Any]]:
    return [{"path": _scored_rollout_path(out_path), "kind": "file"}]


def _analysis_stage_artifacts(out_path: Path) -> list[dict[str, Any]]:
    return [
        {"path": _long_csv_path(out_path), "kind": "file"},
        {"path": _summary_csv_path(out_path), "kind": "file"},
        {"path": _figure_path(out_path), "kind": "file"},
    ]


def _generation_config(config: ExperimentConfig, base_model_ref: str, adapter_path: str | None) -> InferenceConfig:
    return InferenceConfig(
        model=base_model_ref.removeprefix("local://"),
        provider="local",
        generation=config.generation,
        local=LocalProviderConfig(
            dtype="bfloat16",
            device_map="auto",
            adapter_path=adapter_path,
        ),
        output_path=None,
    )


def _run_rollout_stage(
    *,
    config: ExperimentConfig,
    out_path: Path,
    run_id: str,
    config_hash: str,
    constitution: str,
    base_model_ref: str,
    prompt_path: Path,
    rerun: bool,
) -> Path:
    artifacts = _rollout_stage_artifacts(out_path)
    if not rerun and _ensure_stage_available(
        out_path=out_path,
        run_id=run_id,
        stage_name="rollouts",
        config_hash=config_hash,
        artifacts=artifacts,
        hf_repo_id=config.results_hf_repo,
    ):
        return _combined_rollout_path(out_path)

    prompts = _prompt_rows(prompt_path)
    prompt_dataset = Dataset.from_list([{"question": row["question"]} for row in prompts])
    combined_rows: list[dict[str, Any]] = []

    for spec_name, label, adapter_kind in _MODEL_VARIANTS:
        adapter_path = None
        if adapter_kind is not None:
            adapter_path = f"local://{_adapter_dir(out_path, constitution, adapter_kind)}"

        inference_config = _generation_config(
            config=config,
            base_model_ref=base_model_ref,
            adapter_path=adapter_path,
        )
        result_dataset, _ = run_inference(inference_config, dataset=prompt_dataset)
        result_rows = result_dataset.to_list()
        rollout_rows: list[dict[str, Any]] = []
        for idx, (prompt_row, generated_row) in enumerate(zip(prompts, result_rows, strict=True)):
            rollout_rows.append(
                {
                    "sample_id": prompt_row.get("id", idx),
                    "prompt_index": prompt_row.get("prompt_index", idx),
                    "model_spec": spec_name,
                    "model_label": label,
                    "question": prompt_row["question"],
                    "response": generated_row.get("response", ""),
                    "response_index": generated_row.get("response_index", 0),
                }
            )
        _write_jsonl(_variant_rollout_path(out_path, spec_name), rollout_rows)
        combined_rows.extend(rollout_rows)

    _write_jsonl(_combined_rollout_path(out_path), combined_rows)
    _publish_stage(
        out_path=out_path,
        run_id=run_id,
        stage_name="rollouts",
        config_hash=config_hash,
        artifacts=artifacts,
        hf_repo_id=config.results_hf_repo,
    )
    return _combined_rollout_path(out_path)


def _run_scoring_stage(
    *,
    config: ExperimentConfig,
    out_path: Path,
    run_id: str,
    config_hash: str,
    rerun: bool,
) -> Path:
    artifacts = _scoring_stage_artifacts(out_path)
    if not rerun and _ensure_stage_available(
        out_path=out_path,
        run_id=run_id,
        stage_name="scoring",
        config_hash=config_hash,
        artifacts=artifacts,
        hf_repo_id=config.results_hf_repo,
    ):
        return _scored_rollout_path(out_path)

    rollout_rows = _prompt_rows(_combined_rollout_path(out_path))
    metrics_config = PersonaMetricsConfig(
        evaluations=["conscientiousness_v2"],
        judge=config.judge,
        response_column="response",
        question_column="question",
        output_path=_scored_rollout_path(out_path),
    )
    _score_rollout_rows_with_progress(rollout_rows, metrics_config)

    _publish_stage(
        out_path=out_path,
        run_id=run_id,
        stage_name="scoring",
        config_hash=config_hash,
        artifacts=artifacts,
        hf_repo_id=config.results_hf_repo,
    )
    return _scored_rollout_path(out_path)


def _score_rollout_rows_with_progress(
    rollout_rows: list[dict[str, Any]],
    metrics_config: PersonaMetricsConfig,
) -> None:
    """Score rollout rows with periodic progress logging."""
    logger = setup_logging()
    metrics = create_persona_metrics(metrics_config)
    if len(metrics) != 1:
        raise ValueError(f"Expected exactly one metric, got {[metric.name for metric in metrics]}")

    metric = metrics[0]
    responses = [str(row.get("response", "")) for row in rollout_rows]
    questions = [str(row.get("question", "")) for row in rollout_rows]
    total = len(rollout_rows)
    chunk_size = max(metrics_config.judge.max_concurrent * 5, 25)

    logger.info(
        "Running %d persona metric(s) on %d samples: %s",
        1,
        total,
        [metric.name],
    )
    logger.info(
        "Running persona metric: %s (chunk_size=%d, max_concurrent=%d)",
        metric.name,
        chunk_size,
        metrics_config.judge.max_concurrent,
    )

    all_results: list[dict[str, float | int | str]] = []
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        batch_results = asyncio.run(
            metric.evaluate_batch_async(
                responses[start:end],
                questions[start:end],
            )
        )
        all_results.extend(batch_results)
        logger.info(
            "Scoring progress | %d/%d samples (%d%%)",
            end,
            total,
            round((end / total) * 100),
        )

    for row, metric_values in zip(rollout_rows, all_results, strict=True):
        existing = row.get(metrics_config.metrics_key)
        if isinstance(existing, dict):
            row[metrics_config.metrics_key] = {**existing, **metric_values}
        else:
            row[metrics_config.metrics_key] = metric_values

    save_path = Path(metrics_config.output_path) if metrics_config.output_path else None
    if save_path is None:
        raise ValueError("metrics_config.output_path is required for staged scoring output")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(rollout_rows, save_path)
    logger.info("Saved persona metrics output to %s", save_path)
    logger.info("Completed persona metric: %s", metric.name)


def _plot_bar_chart(summary_df: pd.DataFrame, out_path: Path, title: str) -> Path:
    figure_path = _figure_path(out_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [_MODEL_COLORS[label] for label in summary_df["model_label"]]
    ax.bar(summary_df["model_label"], summary_df["mean_conscientiousness"], color=colors)
    ax.set_ylabel("Mean conscientiousness_v2 score")
    ax.set_xlabel("Model variant")
    ax.set_title(title)
    ax.set_ylim(-4.0, 4.0)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    return figure_path


def _run_analysis_stage(
    *,
    config: ExperimentConfig,
    out_path: Path,
    run_id: str,
    config_hash: str,
    rerun: bool,
) -> tuple[Path, Path, Path]:
    artifacts = _analysis_stage_artifacts(out_path)
    if not rerun and _ensure_stage_available(
        out_path=out_path,
        run_id=run_id,
        stage_name="analysis",
        config_hash=config_hash,
        artifacts=artifacts,
        hf_repo_id=config.results_hf_repo,
    ):
        return _long_csv_path(out_path), _summary_csv_path(out_path), _figure_path(out_path)

    scored_rows = _prompt_rows(_scored_rollout_path(out_path))
    for row in scored_rows:
        metrics = row.get("persona_metrics", {})
        row["conscientiousness_v2_score"] = (
            metrics.get("conscientiousness_v2.score")
            if isinstance(metrics, dict)
            else None
        )
        row["conscientiousness_v2_reasoning"] = (
            metrics.get("conscientiousness_v2.reasoning")
            if isinstance(metrics, dict)
            else None
        )

    long_df = pd.DataFrame.from_records(scored_rows)
    long_csv_path = _long_csv_path(out_path)
    long_csv_path.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(long_csv_path, index=False)

    summary_df = (
        long_df.groupby(["model_spec", "model_label"], as_index=False)["conscientiousness_v2_score"]
        .mean()
        .rename(columns={"conscientiousness_v2_score": "mean_conscientiousness"})
    )
    label_order = [label for _, label, _ in _MODEL_VARIANTS]
    summary_df["model_label"] = pd.Categorical(
        summary_df["model_label"],
        categories=label_order,
        ordered=True,
    )
    summary_df = summary_df.sort_values("model_label").reset_index(drop=True)
    summary_csv_path = _summary_csv_path(out_path)
    summary_df.to_csv(summary_csv_path, index=False)

    figure_path = _plot_bar_chart(
        summary_df,
        out_path,
        title=(
            f"Mean conscientiousness_v2: {config.prompt_source.max_samples} prompts "
            f"(seed={config.prompt_source.seed})"
        ),
    )
    _publish_stage(
        out_path=out_path,
        run_id=run_id,
        stage_name="analysis",
        config_hash=config_hash,
        artifacts=artifacts,
        hf_repo_id=config.results_hf_repo,
    )
    return long_csv_path, summary_csv_path, figure_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage-separated OCT conscientiousness evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-hf-repo", required=True)
    parser.add_argument("--source-run-dir", default=None)
    parser.add_argument("--results-hf-repo", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--constitution", default=None)
    parser.add_argument("--base-model-ref", default=None)
    parser.add_argument("--model-cache-root", default=str(_DEFAULT_MODEL_CACHE_ROOT))

    parser.add_argument("--prompt-source", default="huggingface", choices=["huggingface", "local"])
    parser.add_argument("--prompt-dataset-name", default=_DEFAULT_PROMPT_DATASET)
    parser.add_argument("--prompt-split", default=_DEFAULT_PROMPT_SPLIT)
    parser.add_argument("--prompt-path", default=None)
    parser.add_argument("--question-column", default="question")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--sample-seed", type=int, default=223457)

    parser.add_argument("--judge-provider", default="openai", choices=["openai", "openrouter", "anthropic"])
    parser.add_argument("--judge-model", default="gpt-5-nano-2025-08-07")
    parser.add_argument("--judge-api-key-env", default=None)
    parser.add_argument("--judge-max-tokens", type=int, default=10000)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-max-concurrent", type=int, default=10)
    parser.add_argument("--judge-timeout", type=int, default=60)

    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--greedy", action="store_true")

    parser.add_argument(
        "--through-stage",
        default="analysis",
        choices=["rollouts", "scoring", "analysis"],
        help="Run through this stage and then stop.",
    )
    parser.add_argument("--rerun", action="store_true", help="Rerun downstream stages even if cached.")
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        source_hf_repo=args.source_hf_repo,
        source_run_dir=args.source_run_dir,
        results_hf_repo=args.results_hf_repo or args.source_hf_repo,
        constitution=args.constitution,
        base_model_ref=args.base_model_ref,
        model_cache_root=args.model_cache_root,
        prompt_source=PromptSourceConfig(
            source=args.prompt_source,
            name=args.prompt_dataset_name,
            split=args.prompt_split,
            path=args.prompt_path,
            question_column=args.question_column,
            max_samples=args.max_samples,
            seed=args.sample_seed,
        ),
        judge=JudgeLLMConfig(
            provider=args.judge_provider,
            model=args.judge_model,
            api_key_env=args.judge_api_key_env,
            max_tokens=args.judge_max_tokens,
            temperature=args.judge_temperature,
            max_concurrent=args.judge_max_concurrent,
            timeout=args.judge_timeout,
        ),
        generation=GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=0.0 if args.greedy else args.temperature,
            top_p=1.0 if args.greedy else args.top_p,
            do_sample=not args.greedy,
            batch_size=args.batch_size,
            num_responses_per_prompt=1,
        ),
    )


def main() -> None:
    args = _parse_args()
    config = _config_from_args(args)

    source_run_dir = _resolve_source_run_dir(config)
    source_run_config_payload = _load_source_run_config(config, source_run_dir)
    constitution = config.constitution or _default_constitution(source_run_config_payload)
    base_model_ref_raw = config.base_model_ref or _default_base_model_ref(source_run_config_payload)

    run_id = _make_run_id(config, source_run_dir, constitution, base_model_ref_raw)
    out_path = Path(args.out_dir).resolve() if args.out_dir else (_DEFAULT_OUTPUT_ROOT / run_id).resolve()
    config_payload = {
        **config.model_dump(),
        "source_run_dir": source_run_dir,
        "constitution": constitution,
        "base_model_ref": base_model_ref_raw,
        "through_stage": args.through_stage,
    }
    config_hash = _config_hash(config_payload)

    print("OCT conscientiousness staged eval")
    print(f"  source repo:      {config.source_hf_repo}")
    print(f"  source run dir:   {source_run_dir}")
    print(f"  results repo:     {config.results_hf_repo}")
    print(f"  local out dir:    {out_path}")
    print(f"  run id:           {run_id}")

    _ensure_run_config(out_path=out_path, run_id=run_id, config_hash=config_hash, config_payload=config_payload)

    _hydrate_source_artifacts(
        config=config,
        out_path=out_path,
        run_id=run_id,
        config_hash=config_hash,
        source_run_dir=source_run_dir,
        constitution=constitution,
        rerun=False,
    )
    prompt_path = _prepare_prompt_subset(
        config=config,
        out_path=out_path,
        run_id=run_id,
        config_hash=config_hash,
        rerun=False,
    )
    base_model_ref = _ensure_base_model(
        base_model_ref=base_model_ref_raw,
        model_cache_root=Path(config.model_cache_root).expanduser().resolve(),
    )

    rollout_path = _run_rollout_stage(
        config=config,
        out_path=out_path,
        run_id=run_id,
        config_hash=config_hash,
        constitution=constitution,
        base_model_ref=base_model_ref,
        prompt_path=prompt_path,
        rerun=args.rerun,
    )
    print(f"Rollouts:           {rollout_path}")
    if args.through_stage == "rollouts":
        return

    scored_path = _run_scoring_stage(
        config=config,
        out_path=out_path,
        run_id=run_id,
        config_hash=config_hash,
        rerun=args.rerun,
    )
    print(f"Scored rollouts:    {scored_path}")
    if args.through_stage == "scoring":
        return

    long_csv_path, summary_csv_path, figure_path = _run_analysis_stage(
        config=config,
        out_path=out_path,
        run_id=run_id,
        config_hash=config_hash,
        rerun=args.rerun,
    )
    print(f"Long CSV:           {long_csv_path}")
    print(f"Summary CSV:        {summary_csv_path}")
    print(f"Bar chart:          {figure_path}")


if __name__ == "__main__":
    main()
