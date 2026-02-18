#!/usr/bin/env python
"""Run a suite of Inspect AI evaluations on a HuggingFace model.

Evals: Theory of Mind, GSM8K, MMLU, TruthfulQA, PopQA.

Usage:
    uv run python scripts/run_tom_eval.py
    uv run python scripts/run_tom_eval.py --eval tom gsm8k mmlu truthfulqa popqa
    uv run python scripts/run_tom_eval.py --eval popqa --popqa-limit 500
"""

import argparse
import json

from inspect_ai import Task, eval
from inspect_ai.dataset import Sample, MemoryDataset, example_dataset
from inspect_ai.scorer import model_graded_fact, includes
from inspect_ai.solver import chain_of_thought, generate, self_critique

from inspect_evals.gsm8k import gsm8k
from inspect_evals.mmlu.mmlu import mmlu_0_shot
from inspect_evals.truthfulqa import truthfulqa

MODEL = "hf/lukebaines/san-fran-train-20260212-132049"

ALL_EVALS = ["tom", "gsm8k", "mmlu", "truthfulqa", "popqa"]


# ---------- Theory of Mind (built-in example dataset) ----------
def theory_of_mind() -> Task:
    return Task(
        dataset=example_dataset("theory_of_mind"),
        solver=[chain_of_thought(), generate(), self_critique()],
        scorer=model_graded_fact(),
    )


# ---------- PopQA (no pre-built inspect-evals task, load from HF) ----------
def popqa(limit: int = 1000) -> Task:
    from datasets import load_dataset

    ds = load_dataset("akariasai/PopQA", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    samples = []
    for row in ds:
        answers = json.loads(row["possible_answers"])
        samples.append(
            Sample(
                input=row["question"],
                target=answers,
            )
        )

    return Task(
        dataset=MemoryDataset(samples=samples, name="popqa"),
        solver=[generate()],
        scorer=includes(),
    )


def build_tasks(names: list[str], popqa_limit: int) -> list[Task]:
    builders = {
        "tom": theory_of_mind,
        "gsm8k": lambda: gsm8k(fewshot=5),
        "mmlu": lambda: mmlu_0_shot(),
        "truthfulqa": lambda: truthfulqa(target="mc1"),
        "popqa": lambda: popqa(limit=popqa_limit),
    }
    return [builders[n]() for n in names]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evals on a HuggingFace model")
    parser.add_argument(
        "--eval",
        nargs="+",
        choices=ALL_EVALS,
        default=ALL_EVALS,
        help="Which evals to run (default: all)",
    )
    parser.add_argument(
        "--model",
        default=MODEL,
        help=f"Model to evaluate (default: {MODEL})",
    )
    parser.add_argument(
        "--popqa-limit",
        type=int,
        default=10,
        help="Max PopQA samples (default: 10, full dataset is ~14k)",
    )
    args = parser.parse_args()

    tasks = build_tasks(args.eval, args.popqa_limit)

    results = eval(tasks, model=args.model, limit=10)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for log in results:
        name = log.eval.task
        print(f"\n{name}  (status: {log.status})")
        if log.results:
            for metric_name, metric in log.results.metrics.items():
                print(f"  {metric_name}: {metric.value}")


if __name__ == "__main__":
    main()
