"""Add a persona metric as an additional quality judge to a bloom-data directory.

Usage:
    uv run python scripts/experiments/add_bloom_quality.py \\
        --metric BetterCoherenceEvaluation \\
        --quality-name coherence \\
        --bloom-data bloom-data \\
        --max-examples 5

The script reads the rubric and few-shot examples directly from the metric class
in src_dev/persona_metrics, formats them into a bloom-compatible description, and
writes them into bloom-data/behaviors.json + bloom-data/seed.yaml.

To add more qualities later, just run it again with a different --metric/--quality-name.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from pathlib import Path
from typing import Type

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src_dev.persona_metrics.metrics.llm_judge_base import LLMJudgeMetric

# Scores to exclude from few-shot examples (special sentinel values not usable in bloom)
_EXCLUDED_SCORES = {-99999}

# Approximate score range to sample from (inclusive)
_EXAMPLE_SCORE_MIN = 0
_EXAMPLE_SCORE_MAX = 10


def _extract_rubric(template: str) -> str:
    """Return the rubric portion of a template, stripping example/eval placeholders."""
    # Cut off at the first template placeholder or "examples" header
    for marker in ("{examples_text}", "Here are examples", "Examples:\n", "Now evaluate"):
        if marker in template:
            template = template[: template.index(marker)]
    return template.strip()


def _select_examples(
    examples: list[dict],
    max_examples: int | None,
) -> list[dict]:
    """Return usable examples, optionally capped at max_examples spread evenly across the score range."""
    usable = [e for e in examples if e["score"] not in _EXCLUDED_SCORES]
    if not usable:
        return []
    if max_examples is None or len(usable) <= max_examples:
        return usable
    # Evenly spaced indices
    step = (len(usable) - 1) / (max_examples - 1)
    indices = {round(i * step) for i in range(max_examples)}
    return [usable[i] for i in sorted(indices)]


def _format_examples(examples: list[dict]) -> str:
    parts = []
    for i, ex in enumerate(examples, 1):
        parts.append(
            f"Example {i}:\n"
            f"  Question: {ex['question']}\n"
            f"  Response: {ex['response']}\n"
            f"  Score: {ex['score']}\n"
            f"  Reasoning: {ex['reasoning']}"
        )
    return "\n\n".join(parts)


def metric_to_bloom_description(
    metric_cls: Type[LLMJudgeMetric],
    max_examples: int | None = None,
) -> str:
    """Build a bloom behaviors.json description string from an LLMJudgeMetric subclass.

    The returned string is injected into the judge prompt as:
        "Score for {quality_name} (1-10): {description}"

    Args:
        metric_cls: A subclass of LLMJudgeMetric with default_template and default_examples.
        max_examples: How many few-shot examples to embed (spread across score range).

    Returns:
        A self-contained rubric + few-shot examples string.
    """
    rubric = _extract_rubric(metric_cls.default_template)
    examples = _select_examples(metric_cls.default_examples, max_examples)
    examples_text = _format_examples(examples)
    return f"{rubric}\n\nFew-shot examples:\n\n{examples_text}"


def add_quality_to_bloom(
    bloom_data_dir: Path,
    quality_name: str,
    metric_cls: Type[LLMJudgeMetric],
    max_examples: int | None = None,
) -> None:
    """Add a metric as a named additional quality to bloom-data.

    Updates:
      - bloom-data/behaviors.json  — adds/replaces the quality description
      - bloom-data/seed.yaml       — appends quality_name to judge.additional_qualities

    Args:
        bloom_data_dir: Path to the bloom-data directory.
        quality_name: Key name for behaviors.json / seed.yaml (e.g. "coherence").
        metric_cls: LLMJudgeMetric subclass to extract rubric and examples from.
        max_examples: Number of few-shot examples to embed.
    """
    behaviors_path = bloom_data_dir / "behaviors.json"
    seed_path = bloom_data_dir / "seed.yaml"

    # --- behaviors.json ---
    behaviors = json.loads(behaviors_path.read_text())
    description = metric_to_bloom_description(metric_cls, max_examples)
    behaviors[quality_name] = description
    behaviors_path.write_text(json.dumps(behaviors, indent=2) + "\n")
    print(f"[behaviors.json] Added/updated '{quality_name}'")

    # --- seed.yaml additional_qualities ---
    seed_text = seed_path.read_text()
    match = re.search(r'(additional_qualities:\s*\[)([^\]]*)(\])', seed_text)
    if not match:
        print("Warning: could not find additional_qualities in seed.yaml — skipping")
        return

    raw_items = match.group(2)
    existing = [q.strip().strip('"').strip("'") for q in raw_items.split(",") if q.strip().strip('"').strip("'")]
    if quality_name in existing:
        print(f"[seed.yaml]      '{quality_name}' already in additional_qualities")
        return

    existing.append(quality_name)
    new_list = ", ".join(f'"{q}"' for q in existing)
    seed_text = seed_text[: match.start(1)] + f'additional_qualities: [{new_list}]' + seed_text[match.end(3) :]
    seed_path.write_text(seed_text)
    print(f"[seed.yaml]      Added '{quality_name}' to additional_qualities")


def _resolve_metric_cls(metric_name: str) -> Type[LLMJudgeMetric]:
    """Find a metric class by name across known metric modules."""
    modules = [
        "src_dev.persona_metrics.metrics.coherence",
        "src_dev.persona_metrics.metrics.ocean_v2",
        "src_dev.persona_metrics.metrics.ocean",
        "src_dev.persona_metrics.metrics.text_style",
    ]
    for mod_name in modules:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        cls = getattr(mod, metric_name, None)
        if cls is not None:
            return cls
    raise ValueError(
        f"Could not find metric class '{metric_name}' in any known module. "
        "Pass the fully qualified module path with --module if needed."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metric", required=True, help="Metric class name, e.g. BetterCoherenceEvaluation")
    parser.add_argument("--module", default=None, help="Optional fully-qualified module path if class is not auto-found")
    parser.add_argument("--quality-name", required=True, help="Key to use in behaviors.json / seed.yaml")
    parser.add_argument("--bloom-data", default="bloom-data", help="Path to bloom-data directory (default: bloom-data)")
    parser.add_argument("--max-examples", type=int, default=None, help="Max few-shot examples to embed (default: all)")
    args = parser.parse_args()

    if args.module:
        mod = importlib.import_module(args.module)
        metric_cls = getattr(mod, args.metric)
    else:
        metric_cls = _resolve_metric_cls(args.metric)

    bloom_data_dir = Path(args.bloom_data)
    if not bloom_data_dir.is_dir():
        sys.exit(f"Error: bloom-data directory not found: {bloom_data_dir}")

    add_quality_to_bloom(bloom_data_dir, args.quality_name, metric_cls, args.max_examples)


if __name__ == "__main__":
    main()
