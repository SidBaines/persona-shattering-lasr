#!/usr/bin/env python3
"""Coherence judge calibration — graded degradation dataset + judge scoring.

Calibrates the coherence judge (0–100 scale) by generating responses at
five known degradation levels, then scoring them with a judge panel.

Degradation conditions
----------------------
Rather than relying on system prompts (which only produce extreme high/low
signals), this pipeline generates a *neutral* baseline response and then
rewrites it with increasing levels of degradation using an LLM rewriter.
This gives graded examples across the full 0–100 range:

    level_0_baseline    — original coherent response (expected ~85–95)
    level_1_minor       — minor issues: slightly awkward transitions, one
                          tangential sentence (expected ~65–80)
    level_2_moderate    — moderate: ideas out of order, repetition, weak
                          topic relevance (expected ~45–60)
    level_3_severe      — severe: shuffled sentences, missing context,
                          contradictions (expected ~25–40)
    level_4_incoherent  — extreme: scrambled, non-sequiturs, topic drift,
                          word salad fragments (expected ~0–20)

This mirrors the G-Eval methodology of anchoring judge scores to known
quality levels, adapted for our setup (no logprob access needed).

Usage
-----

Generate calibration dataset (rewrite responses at 5 degradation levels)::

    uv run python scripts_dev/persona_metrics/coherence_calibration.py \\
        --stage generate --max-prompts 60

Judge the dataset::

    uv run python scripts_dev/persona_metrics/coherence_calibration.py \\
        --stage judge \\
        --dataset scratch/coherence_calibration/runs/<key>/exports/all_responses.jsonl

Generate + judge in one shot::

    uv run python scripts_dev/persona_metrics/coherence_calibration.py \\
        --stage all --max-prompts 60

Judge an existing dataset with a different rater panel::

    uv run python scripts_dev/persona_metrics/coherence_calibration.py \\
        --stage judge \\
        --dataset scratch/coherence_calibration/runs/<key>/exports/all_responses.jsonl

Dry run::

    uv run python scripts_dev/persona_metrics/coherence_calibration.py --dry-run

What to look for in results
---------------------------

In analysis/condition_metrics.json, you want monotonically decreasing mean
scores: level_0 > level_1 > level_2 > level_3 > level_4.

If the ordering is wrong or the spread is compressed (e.g. everything scores
60–80), the judge prompt or few-shot examples need revision.

Inter-rater agreement (Krippendorff α) should be ≥ 0.7 on coherence;
lower agreement likely means the scoring rubric is ambiguous.

Outputs
-------

    scratch/coherence_calibration/runs/<dataset-key>/
        prompts/source_prompts.jsonl
        responses/{level_0_baseline,...,level_4_incoherent}/
        exports/all_responses.jsonl         ← inspect with jsonl_tui
        judge_runs/<judge-key>/
            judge_calls/raw/<rater_id>.jsonl
            analysis/summary.json
            analysis/condition_metrics.json  ← key: check degradation ordering
            plots/
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src_dev.common.config import DatasetConfig
from src_dev.datasets import load_dataset_from_config
from src_dev.inference import InferenceConfig, run_inference
from src_dev.inference.config import GenerationConfig
from src_dev.persona_metrics.config import JudgeLLMConfig
from src_dev.persona_metrics.llm_judge_agreement import (
    JudgeRaterConfig,
    OceanJudgeRunConfig,
    _judge_paths,
    _run_judge_panel,
    _analyze,
    build_judge_run_fingerprint,
    build_judge_run_key,
    get_judge_run_dir,
)
from src_dev.utils.io import read_jsonl, write_jsonl

# ---------------------------------------------------------------------------
# Degradation system prompts
# ---------------------------------------------------------------------------

# Ordered from most coherent to least — these instruct the LLM to rewrite
# the baseline response at each degradation level.
DEGRADATION_CONDITIONS: dict[str, str] = {
    "level_0_baseline": (
        "You are a helpful assistant. Answer the user's question clearly and coherently."
    ),
    "level_1_minor": (
        "You are a helpful assistant. Answer the user's question, but introduce minor "
        "coherence issues: add one slightly tangential sentence that drifts from the "
        "main point, and use at least one awkward transition. The response should still "
        "be mostly readable and on-topic."
    ),
    "level_2_moderate": (
        "You are a helpful assistant. Answer the user's question, but introduce moderate "
        "coherence problems: present ideas in a slightly illogical order, repeat a point "
        "unnecessarily, and include one sentence that only weakly relates to the question. "
        "The response should feel disorganized but still contain relevant content."
    ),
    "level_3_severe": (
        "You are a helpful assistant. Attempt to answer the user's question, but with "
        "severe coherence problems: shuffle the logical order of ideas, introduce at least "
        "one clear contradiction, include sentences that seem to belong to a different topic, "
        "and leave some ideas without connection to others. The response should be difficult "
        "to follow."
    ),
    "level_4_incoherent": (
        "You are a helpful assistant. Produce a response to the user's question that is "
        "largely incoherent: mix unrelated topics, use non-sequiturs, introduce sentence "
        "fragments, jump abruptly between unrelated ideas, and make it very difficult to "
        "extract any coherent meaning. Some words related to the topic may appear, but the "
        "overall response should make little sense."
    ),
}

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LOCAL_ROOT = Path("scratch/coherence_calibration")
HF_REPO_ID = "persona-shattering-lasr/ocean_judge_runs"
HF_ROOT_PREFIX = "coherence_calibration"


class CoherenceDatasetConfig(BaseModel):
    max_prompts: int = 60
    seed: int = 42
    assistant_inference: InferenceConfig
    prompt_dataset_path: Path = Path("data/assistant-axis-extraction-questions.jsonl")
    local_root_dir: Path = LOCAL_ROOT
    hf_repo_id: str = HF_REPO_ID
    hf_root_prefix: str = HF_ROOT_PREFIX


class CoherenceJudgeRunConfig(BaseModel):
    dataset_path: Path
    judge_raters: list[JudgeRaterConfig]
    judge_repeats: int = 3
    retry_call_errors: bool = True
    plot: bool = True
    hf_repo_id: str = HF_REPO_ID


# ---------------------------------------------------------------------------
# Default panels
# ---------------------------------------------------------------------------

_DEFAULT_ASSISTANT = InferenceConfig(
    provider="openrouter",
    model="openai/gpt-4o-mini",
    generation=GenerationConfig(temperature=0.7),
)

_DEFAULT_RATERS = [
    JudgeRaterConfig(
        rater_id="gpt_4o_mini",
        metric_name="coherence",
        judge=JudgeLLMConfig(
            provider="openrouter",
            model="openai/gpt-4o-mini",
            temperature=0.0,
            max_concurrent=10,
        ),
    ),
    JudgeRaterConfig(
        rater_id="haiku_35",
        metric_name="coherence",
        judge=JudgeLLMConfig(
            provider="openrouter",
            model="anthropic/claude-3.5-haiku",
            temperature=0.0,
            max_concurrent=10,
        ),
    ),
    JudgeRaterConfig(
        rater_id="gemini_flash_20",
        metric_name="coherence",
        judge=JudgeLLMConfig(
            provider="openrouter",
            model="google/gemini-2.0-flash-001",
            temperature=0.0,
            max_concurrent=10,
        ),
    ),
]

# ---------------------------------------------------------------------------
# Fingerprint / paths
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _stable_json(data) -> str:
    def _default(v):
        if isinstance(v, Path):
            return str(v)
        if hasattr(v, "model_dump"):
            return v.model_dump(mode="json")
        raise TypeError(type(v).__name__)
    return json.dumps(data, sort_keys=True, ensure_ascii=False, default=_default)


def build_coherence_dataset_fingerprint(config: CoherenceDatasetConfig) -> str:
    payload = {
        "type": "coherence_calibration",
        "prompt_dataset_path": str(config.prompt_dataset_path),
        "seed": config.seed,
        "max_prompts": config.max_prompts,
        "assistant_inference": {
            "model": config.assistant_inference.model,
            "provider": config.assistant_inference.provider,
            "generation": config.assistant_inference.generation.model_dump(mode="json"),
        },
        "conditions": list(DEGRADATION_CONDITIONS.keys()),
    }
    return hashlib.sha256(_stable_json(payload).encode()).hexdigest()


def build_coherence_dataset_run_key(config: CoherenceDatasetConfig) -> str:
    return f"coherence-seed-{config.seed}-{build_coherence_dataset_fingerprint(config)[:12]}"


def _dataset_paths(config: CoherenceDatasetConfig) -> dict[str, Path]:
    run_dir = config.local_root_dir / "runs" / build_coherence_dataset_run_key(config)
    return {
        "run_dir": run_dir,
        "manifest": run_dir / "manifest.json",
        "prompts_dir": run_dir / "prompts",
        "source_prompts": run_dir / "prompts" / "source_prompts.jsonl",
        "responses_dir": run_dir / "responses",
        "exports_dir": run_dir / "exports",
        "all_responses": run_dir / "exports" / "all_responses.jsonl",
        "judge_runs_dir": run_dir / "judge_runs",
    }


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def _is_condition_complete(condition_dir: Path, expected_rows: int) -> bool:
    from src_dev.datasets import load_samples, resume_state
    if not (condition_dir / "manifest.json").exists():
        return False
    try:
        state = resume_state(condition_dir, "inference", max_attempts=3)
        samples = load_samples(condition_dir)
    except Exception:
        return False
    if state["pending"] or state["terminal"]:
        return False
    if len(samples) != expected_rows:
        return False
    return all(s.inference.status == "success" for s in samples)


def _flatten_responses(paths: dict[str, Path], prompt_rows: list[dict]) -> list[dict]:
    if paths["all_responses"].exists():
        return read_jsonl(paths["all_responses"])

    from src_dev.datasets import load_samples
    prompt_by_index = {i: row for i, row in enumerate(prompt_rows)}
    rows = []
    for condition_name in DEGRADATION_CONDITIONS:
        condition_dir = paths["responses_dir"] / condition_name
        samples = load_samples(condition_dir)
        for sample in samples:
            user_msgs = [m.content for m in sample.messages if m.role == "user"]
            asst_msgs = [m.content for m in sample.messages if m.role == "assistant"]
            row_index = int(sample.source_info.get("row_index", -1))
            prompt_row = prompt_by_index.get(row_index, {})
            rows.append({
                "response_id": f"{condition_name}:{sample.sample_id}",
                "condition": condition_name,
                "degradation_level": int(condition_name.split("_")[1]),
                "sample_id": sample.sample_id,
                "input_group_id": sample.input_group_id or sample.sample_id,
                "response_index": sample.response_index,
                "prompt_row_index": row_index,
                "prompt_id": prompt_row.get("id", row_index),
                "question": user_msgs[-1] if user_msgs else "",
                "response": asst_msgs[-1] if asst_msgs else "",
                "assistant_model": sample.inference.model,
                "assistant_provider": sample.inference.provider,
                "system_prompt_ref": sample.input.system_prompt_ref,
            })
    write_jsonl(rows, paths["all_responses"])
    return rows


def generate_coherence_dataset(config: CoherenceDatasetConfig) -> Path:
    """Generate the coherence calibration dataset at 5 degradation levels.

    Args:
        config: Dataset generation configuration.

    Returns:
        Path to the frozen ``all_responses.jsonl``.
    """
    paths = _dataset_paths(config)
    for key in ["run_dir", "prompts_dir", "responses_dir", "exports_dir", "judge_runs_dir"]:
        paths[key].mkdir(parents=True, exist_ok=True)

    run_key = build_coherence_dataset_run_key(config)
    paths["manifest"].write_text(
        _stable_json({
            "run_key": run_key,
            "created_at": _now_iso(),
            "stage": "generate",
            "type": "coherence_calibration",
            "conditions": list(DEGRADATION_CONDITIONS.keys()),
        }) + "\n",
        encoding="utf-8",
    )

    # Source prompts
    if not paths["source_prompts"].exists():
        dataset = load_dataset_from_config(
            DatasetConfig(
                source="local",
                path=str(config.prompt_dataset_path),
                max_samples=config.max_prompts,
                seed=config.seed,
            )
        )
        write_jsonl(dataset.to_list(), paths["source_prompts"])

    prompt_rows = read_jsonl(paths["source_prompts"])

    # One response per prompt per condition (no need for multiple responses/prompt
    # since we're calibrating scale coverage, not sampling variance)
    expected_rows = len(prompt_rows)

    for condition_name, system_prompt in DEGRADATION_CONDITIONS.items():
        condition_dir = paths["responses_dir"] / condition_name
        if not _is_condition_complete(condition_dir, expected_rows):
            inference_cfg = config.assistant_inference.model_copy(deep=True)
            inference_cfg.dataset = DatasetConfig(
                source="local",
                path=str(paths["source_prompts"]),
            )
            inference_cfg.run_dir = condition_dir
            inference_cfg.output_path = None
            inference_cfg.system_prompt = system_prompt
            inference_cfg.generation.num_responses_per_prompt = 1
            print(f"  generating {condition_name} ...")
            run_inference(inference_cfg)

    response_rows = _flatten_responses(paths, prompt_rows)

    print(f"  dataset run_key : {run_key}")
    print(f"  responses       : {len(response_rows)}")
    print(f"  all_responses   : {paths['all_responses']}")
    return paths["all_responses"]


# ---------------------------------------------------------------------------
# Judge stage — reuse OceanJudgeRunConfig machinery directly
# ---------------------------------------------------------------------------

def run_coherence_judge_run(
    dataset_path: Path,
    raters: list[JudgeRaterConfig],
    judge_repeats: int = 3,
    retry_call_errors: bool = True,
    plot: bool = True,
) -> dict:
    """Score a coherence calibration dataset with a judge panel.

    Reuses the OceanJudgeRunConfig infrastructure directly — the judge
    pipeline is trait-agnostic, coherence is just another metric name.

    Args:
        dataset_path: Path to all_responses.jsonl.
        raters: Judge rater panel (metric_name should be "coherence").
        judge_repeats: Scoring repeats per response per rater.
        retry_call_errors: Whether to retry failed calls.
        plot: Whether to generate plots.

    Returns:
        Dict with judge_key, judge_dir, and analysis summary.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Reuse OceanJudgeRunConfig with a placeholder trait — trait is only used
    # for system prompt building in generate_ocean_dataset, not in judge scoring.
    from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait
    from src_dev.persona_metrics.llm_judge_agreement import (
        build_judge_run_key,
        _judge_paths,
        _run_judge_panel,
        _analyze,
    )

    config = OceanJudgeRunConfig(
        trait=OceanTrait.neuroticism,  # placeholder, unused during judge scoring
        dataset_path=dataset_path,
        judge_raters=raters,
        judge_repeats=judge_repeats,
        retry_call_errors=retry_call_errors,
        plot=plot,
    )

    paths = _judge_paths(config)
    for key in ["judge_dir", "judge_calls_dir", "judge_raw_dir", "analysis_dir", "plots_dir"]:
        paths[key].mkdir(parents=True, exist_ok=True)

    judge_key = build_judge_run_key(config)
    paths["manifest"].write_text(
        _stable_json({
            "judge_key": judge_key,
            "created_at": _now_iso(),
            "stage": "judge",
            "type": "coherence_calibration",
            "dataset_path": str(dataset_path),
        }) + "\n",
        encoding="utf-8",
    )

    response_rows = read_jsonl(dataset_path)
    progress = asyncio.run(_run_judge_panel(config, response_rows, paths))
    analysis = _analyze(config, paths, response_rows)

    return {
        "judge_key": judge_key,
        "judge_dir": str(paths["judge_dir"]),
        "dataset_path": str(dataset_path),
        "num_responses": len(response_rows),
        "judge_progress": progress,
        "analysis": analysis,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_dry_run(config: CoherenceDatasetConfig, judge_repeats: int, raters: list) -> None:
    run_key = build_coherence_dataset_run_key(config)
    total_responses = config.max_prompts * len(DEGRADATION_CONDITIONS)
    total_calls = total_responses * judge_repeats * len(raters)
    print(f"\n{'=' * 70}")
    print("DRY RUN — coherence judge calibration")
    print(f"{'=' * 70}")
    print(f"  Dataset run key  : {run_key}")
    print(f"  Prompts          : {config.max_prompts}")
    print(f"  Conditions       : {list(DEGRADATION_CONDITIONS.keys())}")
    print(f"  Judge repeats    : {judge_repeats}")
    print(f"  Raters           : {[r.rater_id for r in raters]}")
    print(f"  Total responses  : ~{total_responses}")
    print(f"  Total judge calls: ~{total_calls}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Coherence judge calibration harness (graded degradation).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--stage", choices=["generate", "judge", "all"], default="all")
    parser.add_argument("--dataset", type=Path, default=None,
                        help="Path to existing all_responses.jsonl (skips generation).")
    parser.add_argument("--max-prompts", type=int, default=60)
    parser.add_argument("--judge-repeats", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-haiku", action="store_true",
                        help="Exclude the haiku_35 rater.")
    parser.add_argument("--upload", action="store_true",
                        help="Upload results to HF after judging.")
    args = parser.parse_args()

    if args.dataset and args.stage == "generate":
        parser.error("--dataset cannot be used with --stage generate")

    load_dotenv()

    dataset_config = CoherenceDatasetConfig(
        max_prompts=args.max_prompts,
        assistant_inference=_DEFAULT_ASSISTANT,
    )
    raters = [r for r in _DEFAULT_RATERS if not (args.no_haiku and r.rater_id == "haiku_35")]

    if args.dry_run:
        _print_dry_run(dataset_config, args.judge_repeats, raters)
        return

    dataset_path = args.dataset
    if args.stage in {"generate", "all"} and dataset_path is None:
        print("\nGenerating coherence calibration dataset ...")
        dataset_path = generate_coherence_dataset(dataset_config)

    if args.stage in {"judge", "all"}:
        if dataset_path is None:
            parser.error("--dataset is required for --stage judge")
        print(f"\nRunning coherence judge panel ...")
        result = run_coherence_judge_run(
            dataset_path,
            raters=raters,
            judge_repeats=args.judge_repeats,
        )
        print(f"\n  judge_key  : {result['judge_key']}")
        print(f"  judge_dir  : {result['judge_dir']}")
        print(f"  responses  : {result['num_responses']}")
        if "analysis" in result:
            agreement = result["analysis"].get("agreement", {})
            alpha = agreement.get("ordinal_krippendorff_alpha", float("nan"))
            qwk = agreement.get("mean_pairwise_qwk", float("nan"))
            print(f"  Krippendorff α    : {alpha:.3f}")
            print(f"  Mean pairwise QWK : {qwk:.3f}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
