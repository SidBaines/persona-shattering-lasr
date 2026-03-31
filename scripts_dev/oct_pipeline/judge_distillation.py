"""Score OCT distillation data with an OCEAN LLM judge.

Evaluates both teacher (chosen) and student (rejected) responses from a
distillation JSONL file, producing per-response scores and aggregate
statistics. Useful for validating training data quality before DPO training.

Usage:
    # Score A-minus distillation data with default judge (anthropic/claude-sonnet-4-20250514)
    uv run python scripts_dev/oct_pipeline/judge_distillation.py \
        --config scripts_dev/oct_pipeline/ocean/judge_configs/agreeableness_low.py

    # Override judge model
    uv run python scripts_dev/oct_pipeline/judge_distillation.py \
        --config scripts_dev/oct_pipeline/ocean/judge_configs/agreeableness_low.py \
        --judge-model gpt-4o --judge-provider openai

    # Limit to N samples for a quick check
    uv run python scripts_dev/oct_pipeline/judge_distillation.py \
        --config scripts_dev/oct_pipeline/ocean/judge_configs/agreeableness_low.py \
        --max-samples 20

Outputs:
    <output_dir>/
        scored_responses.jsonl   — per-response scores and reasoning
        summary.json             — aggregate stats (mean, std, distribution)
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

from dotenv import load_dotenv

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

load_dotenv()

from src_dev.persona_metrics.config import JudgeLLMConfig
from src_dev.persona_metrics.metrics.ocean_v2 import (
    AgreeablenessV2Evaluation,
    ConscientiousnessV2Evaluation,
    OceanJudgeV2,
)


# ---------------------------------------------------------------------------
# Registry of available judges
# ---------------------------------------------------------------------------
JUDGE_REGISTRY: dict[str, type[OceanJudgeV2]] = {
    "agreeableness_v2": AgreeablenessV2Evaluation,
    "conscientiousness_v2": ConscientiousnessV2Evaluation,
}


def _load_config_module(path: str) -> ModuleType:
    """Import a Python config file as a module."""
    spec = importlib.util.spec_from_file_location("_judge_config", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


async def _score_one(
    judge: OceanJudgeV2,
    question: str,
    response: str,
) -> tuple[int, str]:
    """Score a single response, returning (score, reasoning)."""
    try:
        score, reasoning = await judge._judge_one(response, question)
        return score, reasoning
    except Exception as e:
        return judge.score_error, f"Error: {e}"


async def score_distillation_data(
    data_path: Path,
    judge: OceanJudgeV2,
    output_dir: Path,
    max_samples: int | None = None,
    student_column: str = "llama-3.1-8b-it",
) -> None:
    """Score teacher and student responses from a distillation JSONL file."""
    # Load data
    with open(data_path) as f:
        rows = [json.loads(line) for line in f]

    if max_samples:
        rows = rows[:max_samples]

    print(f"Scoring {len(rows)} distillation pairs...")
    print(f"  Judge: {judge._judge_config.provider}/{judge._judge_config.model}")
    print(f"  Output: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    scored_path = output_dir / "scored_responses.jsonl"
    summary_path = output_dir / "summary.json"

    teacher_scores = []
    student_scores = []
    scored_rows = []

    for i, row in enumerate(rows):
        question = row["prompt"]
        teacher_response = row["response"]
        student_response = row.get(student_column, "")

        # Score teacher (chosen)
        t_score, t_reasoning = await _score_one(judge, question, teacher_response)
        # Score student (rejected)
        s_score, s_reasoning = await _score_one(judge, question, student_response)

        teacher_scores.append(t_score)
        student_scores.append(s_score)

        scored_row = {
            "question": question,
            "teacher_score": t_score,
            "teacher_reasoning": t_reasoning,
            "student_score": s_score,
            "student_reasoning": s_reasoning,
            "score_gap": t_score - s_score,
        }
        scored_rows.append(scored_row)

        if (i + 1) % 10 == 0 or i == len(rows) - 1:
            t_mean = sum(teacher_scores) / len(teacher_scores)
            s_mean = sum(student_scores) / len(student_scores)
            print(f"  [{i+1}/{len(rows)}] teacher_mean={t_mean:.2f} student_mean={s_mean:.2f}")

    # Write scored responses
    with open(scored_path, "w") as f:
        for row in scored_rows:
            f.write(json.dumps(row) + "\n")

    # Compute summary statistics
    def _stats(scores: list[int]) -> dict:
        valid = [s for s in scores if s != -99]
        if not valid:
            return {"mean": 0, "std": 0, "n": 0, "errors": len(scores)}
        mean = sum(valid) / len(valid)
        std = (sum((s - mean) ** 2 for s in valid) / len(valid)) ** 0.5
        distribution = {str(s): valid.count(s) for s in range(-4, 5) if valid.count(s) > 0}
        return {
            "mean": round(mean, 3),
            "std": round(std, 3),
            "n": len(valid),
            "errors": len(scores) - len(valid),
            "distribution": distribution,
        }

    gaps = [r["score_gap"] for r in scored_rows if r["teacher_score"] != -99 and r["student_score"] != -99]
    gap_mean = sum(gaps) / len(gaps) if gaps else 0

    summary = {
        "data_path": str(data_path),
        "judge": f"{judge._judge_config.provider}/{judge._judge_config.model}",
        "n_samples": len(rows),
        "teacher_stats": _stats(teacher_scores),
        "student_stats": _stats(student_scores),
        "gap_stats": {
            "mean": round(gap_mean, 3),
            "n": len(gaps),
        },
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Summary ===")
    print(f"  Teacher: mean={summary['teacher_stats']['mean']:.2f}, std={summary['teacher_stats']['std']:.2f}")
    print(f"  Student: mean={summary['student_stats']['mean']:.2f}, std={summary['student_stats']['std']:.2f}")
    print(f"  Gap (teacher - student): {gap_mean:.2f}")
    print(f"  Teacher distribution: {summary['teacher_stats'].get('distribution', {})}")
    print(f"  Student distribution: {summary['student_stats'].get('distribution', {})}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score OCT distillation data with an OCEAN LLM judge panel.",
    )
    parser.add_argument("--config", required=True, help="Path to a Python config file")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit to N samples")
    parser.add_argument("--judge-provider", default=None, help="Override judge provider (single-judge mode)")
    parser.add_argument("--judge-model", default=None, help="Override judge model (single-judge mode)")
    args = parser.parse_args()

    config = _load_config_module(args.config)

    # Required config attributes
    data_path = Path(config.DATA_PATH)
    judge_name = config.JUDGE_NAME
    output_dir = Path(config.OUTPUT_DIR)

    # Optional config attributes
    student_column = getattr(config, "STUDENT_COLUMN", "llama-3.1-8b-it")

    # Instantiate judge class
    judge_cls = JUDGE_REGISTRY.get(judge_name)
    if judge_cls is None:
        raise ValueError(f"Unknown judge: {judge_name}. Available: {list(JUDGE_REGISTRY.keys())}")

    # Determine judge configs: panel (JUDGE_CONFIGS dict) or single (JUDGE_CONFIG)
    judge_configs: dict[str, JudgeLLMConfig] = {}

    if args.judge_provider or args.judge_model:
        # CLI override → single judge mode
        cfg = getattr(config, "JUDGE_CONFIG", JudgeLLMConfig())
        if args.judge_provider:
            cfg = cfg.model_copy(update={"provider": args.judge_provider})
        if args.judge_model:
            cfg = cfg.model_copy(update={"model": args.judge_model})
        judge_configs["cli_override"] = cfg
    elif hasattr(config, "JUDGE_CONFIGS"):
        judge_configs = config.JUDGE_CONFIGS
    else:
        judge_configs["default"] = getattr(config, "JUDGE_CONFIG", JudgeLLMConfig())

    # Run each judge
    for rater_id, judge_config in judge_configs.items():
        print(f"\n{'='*60}")
        print(f"  Judge: {rater_id} ({judge_config.provider}/{judge_config.model})")
        print(f"{'='*60}")

        judge = judge_cls(judge_config=judge_config)
        rater_output_dir = Path(output_dir) / rater_id

        asyncio.run(
            score_distillation_data(
                data_path=data_path,
                judge=judge,
                output_dir=rater_output_dir,
                max_samples=args.max_samples,
                student_column=student_column,
            )
        )


if __name__ == "__main__":
    main()
