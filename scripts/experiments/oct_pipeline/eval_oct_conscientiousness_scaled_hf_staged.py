"""Stage-separated OCT conscientiousness evaluation for scaled single adapters."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd
import torch
from datasets import Dataset
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.experiments.oct_pipeline.eval_oct_conscientiousness_hf import (
    _config_hash,
    _default_base_model_ref,
    _default_constitution,
    _ensure_base_model,
    _ensure_run_config,
    _ensure_stage_available,
    _hydrate_source_artifacts,
    _load_source_run_config,
    _prepare_prompt_subset,
    _publish_stage,
    _read_json,
    _resolve_source_run_dir,
    _write_jsonl,
)
from scripts.experiments.oct_pipeline.eval_oct_conscientiousness_hf_staged import (
    PromptSourceConfig,
    _analysis_stage_artifacts,
    _figure_path,
    _long_csv_path,
    _prompt_rows,
    _score_rollout_rows_with_progress,
    _scored_rollout_path,
    _summary_csv_path,
)
from src_dev.common.config import GenerationConfig
from src_dev.inference import InferenceConfig, LocalProviderConfig, run_inference
from src_dev.persona_metrics import JudgeLLMConfig, PersonaMetricsConfig
from src_dev.utils.lora_composition import WeightedAdapter, load_and_scale_adapters

matplotlib.use("Agg")
import matplotlib.pyplot as plt

load_dotenv()


_DEFAULT_OUTPUT_ROOT = Path("scratch/oct_pipeline_eval_runs")
_DEFAULT_MODEL_CACHE_ROOT = Path("/root/.cache/models")
_DEFAULT_RUN_PREFIX = "oct-conscientiousness-scaled"
_DEFAULT_PROMPT_DATASET = "mirlab/TRAIT"
_DEFAULT_PROMPT_SPLIT = "Conscientiousness"

_SCALED_VARIANTS: tuple[tuple[str, str, str | None, float | None], ...] = (
    ("base", "No LoRA", None, None),
    ("dpo_pos2", "DPO x2.0", "dpo", 2.0),
    ("dpo_pos3", "DPO x3.0", "dpo", 3.0),
    ("dpo_neg1", "DPO x-1.0", "dpo", -1.0),
    ("sft_pos2", "SFT x2.0", "sft", 2.0),
    ("sft_pos3", "SFT x3.0", "sft", 3.0),
    ("sft_neg1", "SFT x-1.0", "sft", -1.0),
)

_MODEL_COLORS = {
    "No LoRA": "#37474f",
    "DPO x2.0": "#1565c0",
    "DPO x3.0": "#1e88e5",
    "DPO x-1.0": "#90caf9",
    "SFT x2.0": "#2e7d32",
    "SFT x3.0": "#43a047",
    "SFT x-1.0": "#a5d6a7",
}


class ExperimentConfig(BaseModel):
    """Semantic config for scaled staged evaluation."""

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
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            batch_size=8,
            num_responses_per_prompt=1,
        )
    )


def _short_hash(text: str, length: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def _make_run_id(config: ExperimentConfig, source_run_dir: str, constitution: str, base_model_ref: str) -> str:
    payload = {
        **config.model_dump(),
        "source_run_dir": source_run_dir,
        "constitution": constitution,
        "base_model_ref": base_model_ref,
        "variants": list(_SCALED_VARIANTS),
    }
    digest = _short_hash(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return (
        f"{_DEFAULT_RUN_PREFIX}-{source_run_dir.replace('_', '-')}-"
        f"n{config.prompt_source.max_samples}-s{config.prompt_source.seed}-{digest}"
    )


def _adapter_dir(out_path: Path, constitution: str, kind: str) -> Path:
    return out_path / "source_oct_run" / "lora" / f"{constitution}-{kind}"


def _rollouts_dir(out_path: Path) -> Path:
    return out_path / "rollouts_scaled"


def _variant_rollout_path(out_path: Path, model_spec: str) -> Path:
    return _rollouts_dir(out_path) / f"{model_spec}.jsonl"


def _combined_rollout_path(out_path: Path) -> Path:
    return _rollouts_dir(out_path) / "combined_rollouts.jsonl"


def _rollout_stage_artifacts(out_path: Path) -> list[dict[str, Any]]:
    artifacts = [{"path": _combined_rollout_path(out_path), "kind": "file"}]
    for spec_name, _, _, _ in _SCALED_VARIANTS:
        artifacts.append({"path": _variant_rollout_path(out_path, spec_name), "kind": "file"})
    return artifacts


def _load_scaled_preloaded_model(
    *,
    base_model_path: str,
    adapter_path: Path,
    scale: float,
) -> tuple[Any, Any]:
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            base_model.resize_token_embeddings(len(tokenizer))
    peft_model, _, _ = load_and_scale_adapters(
        base_model,
        adapters=[WeightedAdapter(path=f"local://{adapter_path}", scale=scale)],
    )
    peft_model.eval()
    return peft_model, tokenizer


def _cleanup_preloaded_model(model: Any) -> None:
    try:
        if hasattr(model, "cpu"):
            model.cpu()
    except Exception:
        pass
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
    base_model_path = base_model_ref.removeprefix("local://")

    for spec_name, label, adapter_kind, scale in _SCALED_VARIANTS:
        print(f"  Generating rollouts for {label}")
        local_cfg = LocalProviderConfig(dtype="bfloat16", device_map="auto")
        preloaded_model = None
        if adapter_kind is not None and scale is not None:
            adapter_path = _adapter_dir(out_path, constitution, adapter_kind)
            print(f"    adapter={adapter_path} scale={scale}")
            preloaded_model = _load_scaled_preloaded_model(
                base_model_path=base_model_path,
                adapter_path=adapter_path,
                scale=scale,
            )
            local_cfg.preloaded_model = preloaded_model

        inference_config = InferenceConfig(
            model=base_model_path,
            provider="local",
            generation=config.generation,
            local=local_cfg,
            output_path=None,
        )
        try:
            result_dataset, _ = run_inference(inference_config, dataset=prompt_dataset)
        finally:
            if preloaded_model is not None:
                _cleanup_preloaded_model(preloaded_model[0])

        rollout_rows: list[dict[str, Any]] = []
        for idx, (prompt_row, generated_row) in enumerate(zip(prompts, result_dataset.to_list(), strict=True)):
            rollout_rows.append(
                {
                    "sample_id": prompt_row.get("id", idx),
                    "prompt_index": prompt_row.get("prompt_index", idx),
                    "model_spec": spec_name,
                    "model_label": label,
                    "adapter_kind": adapter_kind,
                    "adapter_scale": scale,
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
    artifacts = [{"path": _scored_rollout_path(out_path), "kind": "file"}]
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


def _plot_bar_chart(summary_df: pd.DataFrame, out_path: Path, title: str) -> Path:
    figure_path = _figure_path(out_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [_MODEL_COLORS[label] for label in summary_df["model_label"]]
    ax.bar(summary_df["model_label"], summary_df["mean_conscientiousness"], color=colors)
    ax.set_ylabel("Mean conscientiousness_v2 score")
    ax.set_xlabel("Model variant")
    ax.set_title(title)
    ax.set_ylim(-4.0, 4.0)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
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
            metrics.get("conscientiousness_v2.score") if isinstance(metrics, dict) else None
        )
        row["conscientiousness_v2_reasoning"] = (
            metrics.get("conscientiousness_v2.reasoning") if isinstance(metrics, dict) else None
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
    label_order = [label for _, label, _, _ in _SCALED_VARIANTS]
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
            f"Scaled adapter conscientiousness_v2: {config.prompt_source.max_samples} prompts "
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
        description="Stage-separated OCT conscientiousness evaluation for scaled DPO/SFT adapters.",
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
    parser.add_argument("--judge-max-tokens", type=int, default=4096)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-max-concurrent", type=int, default=10)
    parser.add_argument("--judge-timeout", type=int, default=60)

    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--greedy", action="store_true")

    parser.add_argument("--through-stage", default="analysis", choices=["rollouts", "scoring", "analysis"])
    parser.add_argument("--rerun", action="store_true")
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
        "variants": list(_SCALED_VARIANTS),
    }
    config_hash = _config_hash(config_payload)

    print("OCT conscientiousness scaled staged eval")
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
