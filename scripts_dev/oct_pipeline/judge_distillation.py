"""Score OCT distillation data with OCEAN LLM judges.

Evaluates both teacher (chosen) and student (rejected) responses from a
distillation JSONL file across one or more OCEAN traits, using a panel of
LLM judges. Useful for:
  - Validating training data quality before DPO training
  - Checking whether targeting one trait bleeds into other OCEAN dimensions
  - Comparing teacher model baseline psychometrics vs student model

Usage:
    # Score on all 5 OCEAN traits with the judge panel from the config
    uv run python scripts_dev/oct_pipeline/judge_distillation.py \
        --config scripts_dev/oct_pipeline/ocean/judge_configs/agreeableness_low.py

    # Score all traits with --all-ocean flag (overrides config JUDGE_NAME)
    uv run python scripts_dev/oct_pipeline/judge_distillation.py \
        --config scripts_dev/oct_pipeline/ocean/judge_configs/agreeableness_low.py \
        --all-ocean

    # Quick check on a subset
    uv run python scripts_dev/oct_pipeline/judge_distillation.py \
        --config scripts_dev/oct_pipeline/ocean/judge_configs/agreeableness_low.py \
        --max-samples 20

Outputs:
    <output_dir>/
        <trait>/
            <rater_id>/
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
    ExtraversionV2Evaluation,
    NeuroticismV2Evaluation,
    OpennessV2Evaluation,
    OceanJudgeV2,
)


# ---------------------------------------------------------------------------
# Registry of available judges
# ---------------------------------------------------------------------------
JUDGE_REGISTRY: dict[str, type[OceanJudgeV2]] = {
    "agreeableness_v2": AgreeablenessV2Evaluation,
    "conscientiousness_v2": ConscientiousnessV2Evaluation,
    "extraversion_v2": ExtraversionV2Evaluation,
    "neuroticism_v2": NeuroticismV2Evaluation,
    "openness_v2": OpennessV2Evaluation,
}

ALL_OCEAN_JUDGES = list(JUDGE_REGISTRY.keys())


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
    timeout_seconds: int = 120,
) -> tuple[int, str]:
    """Score a single response, returning (score, reasoning)."""
    try:
        score, reasoning = await asyncio.wait_for(
            judge._judge_one(response, question),
            timeout=timeout_seconds,
        )
        return score, reasoning
    except asyncio.TimeoutError:
        return judge.score_error, "Error: timeout"
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
            t_valid = [s for s in teacher_scores if s != -99]
            s_valid = [s for s in student_scores if s != -99]
            t_mean = sum(t_valid) / len(t_valid) if t_valid else 0
            s_mean = sum(s_valid) / len(s_valid) if s_valid else 0
            errors = len(teacher_scores) - len(t_valid) + len(student_scores) - len(s_valid)
            err_str = f" ({errors} errors)" if errors else ""
            print(f"  [{i+1}/{len(rows)}] teacher_mean={t_mean:.2f} student_mean={s_mean:.2f}{err_str}")

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
        description="Score OCT distillation data with OCEAN LLM judge(s).",
    )
    parser.add_argument("--config", required=True, help="Path to a Python config file")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit to N samples")
    parser.add_argument("--judge-provider", default=None, help="Override judge provider (single-judge mode)")
    parser.add_argument("--judge-model", default=None, help="Override judge model (single-judge mode)")
    parser.add_argument("--all-ocean", action="store_true",
                        help="Score on all 5 OCEAN traits (overrides JUDGE_NAME in config)")
    args = parser.parse_args()

    config = _load_config_module(args.config)

    # Required config attributes
    data_path = Path(config.DATA_PATH)
    output_dir = Path(config.OUTPUT_DIR)

    # Optional config attributes
    student_column = getattr(config, "STUDENT_COLUMN", "llama-3.1-8b-it")

    # Determine which OCEAN traits to judge
    if args.all_ocean:
        judge_names = ALL_OCEAN_JUDGES
    else:
        judge_name = config.JUDGE_NAME
        if judge_name == "all":
            judge_names = ALL_OCEAN_JUDGES
        else:
            judge_names = [judge_name]

    # Determine judge LLM configs: panel (JUDGE_CONFIGS dict) or single (JUDGE_CONFIG)
    judge_llm_configs: dict[str, JudgeLLMConfig] = {}

    if args.judge_provider or args.judge_model:
        cfg = getattr(config, "JUDGE_CONFIG", JudgeLLMConfig())
        if args.judge_provider:
            cfg = cfg.model_copy(update={"provider": args.judge_provider})
        if args.judge_model:
            cfg = cfg.model_copy(update={"model": args.judge_model})
        judge_llm_configs["cli_override"] = cfg
    elif hasattr(config, "JUDGE_CONFIGS"):
        judge_llm_configs = config.JUDGE_CONFIGS
    else:
        judge_llm_configs["default"] = getattr(config, "JUDGE_CONFIG", JudgeLLMConfig())

    # Build all tasks, then run in a single event loop
    async def _run_all() -> None:
        for trait_judge_name in judge_names:
            judge_cls = JUDGE_REGISTRY.get(trait_judge_name)
            if judge_cls is None:
                print(f"WARNING: Unknown judge '{trait_judge_name}', skipping.")
                continue

            trait_label = trait_judge_name.replace("_v2", "")

            for rater_id, judge_llm_config in judge_llm_configs.items():
                print(f"\n{'='*60}")
                print(f"  Trait: {trait_label} | Judge: {rater_id} ({judge_llm_config.model})")
                print(f"{'='*60}")

                judge = judge_cls(judge_config=judge_llm_config)
                rater_output_dir = Path(output_dir) / trait_label / rater_id

                await score_distillation_data(
                    data_path=data_path,
                    judge=judge,
                    output_dir=rater_output_dir,
                    max_samples=args.max_samples,
                    student_column=student_column,
                )

    asyncio.run(_run_all())


if __name__ == "__main__":
    main()
