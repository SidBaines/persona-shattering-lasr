#!/usr/bin/env python3
"""Prompt iteration script for openness editing variants (O- suppressor).

Supports a tight iteration loop for refining editing prompt templates.

Recommended workflow:

    Step 1 — Scout run (inference + scoring only, no editing):
    Omit --prompts to stop after scoring and inspect the score distribution.

    uv run python scripts/experiments/prompt-iterations/openness_iteration.py \\
        --run-dir scratch/experiments/o_minus/attempt_01/iter_001 \\
        --max-samples 50

    Step 2 — Pick a threshold from the printed distribution, then edit:

    uv run python scripts/experiments/prompt-iterations/openness_iteration.py \\
        --run-dir scratch/experiments/o_minus/attempt_01/iter_001 \\
        --prompts o- neutral_paraphrase_control \\
        --min-openness-score 4

    Step 3 — Iterate on prompt variants (inference, scoring, and existing edits skipped):

    uv run python scripts/experiments/prompt-iterations/openness_iteration.py \\
        --run-dir scratch/experiments/o_minus/attempt_01/iter_001 \\
        --prompts o-v2 \\
        --min-openness-score 4

    Step 4 — Review side by side (command printed at end of run):

    uv run python scripts/jsonl_tui/cli.py \\
        scratch/experiments/o_minus/attempt_01/iter_001/compare.jsonl \\
        --variant-fields original neutral_paraphrase_control o- o-v2 \\
        --meta-fields original_openness_score

Run-dir structure:
    <run-dir>/
    ├── original_responses.jsonl      # inference output; written once, then immutable
    ├── original_scored.jsonl         # openness scores for originals; written once
    ├── edits/
    │   ├── filtered_out.jsonl        # rows below --min-openness-score (written once)
    │   ├── o-.jsonl                  # one file per editing variant
    │   ├── o-v2.jsonl
    │   └── ...
    └── compare.jsonl                 # rebuilt on every run; use for TUI review
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from scripts.common.config import DatasetConfig, GenerationConfig
from scripts.datasets import load_dataset_from_config
from scripts.editing import EditingConfig, QualityConfig, run_editing
from scripts.inference import InferenceConfig, run_inference
from scripts.persona_metrics import PersonaMetricsConfig, run_persona_metrics
from scripts.utils import read_jsonl, write_jsonl

load_dotenv()

DEFAULT_DATASET = "liweijiang/infinite-chats-taxonomy"
DEFAULT_INFERENCE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_INFERENCE_PROVIDER = "openrouter"
DEFAULT_EDITING_MODEL = "gpt-4o-mini"
DEFAULT_EDITING_PROVIDER = "openai"
DEFAULT_MAX_SAMPLES = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iterate on openness editing prompt variants (O- suppressor).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Directory for storing run outputs. Created if it doesn't exist.",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=None,
        metavar="TEMPLATE",
        help=(
            "Editing prompt template names to apply "
            "(e.g. o- o-v2 neutral_paraphrase_control). "
            "If omitted, the script stops after scoring and prints the distribution "
            "so you can pick a --min-openness-score threshold before editing. "
            "Already-existing variants are skipped unless --overwrite-edits is set."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help=f"Number of samples to generate via inference (default: {DEFAULT_MAX_SAMPLES}).",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"HuggingFace dataset name for inference (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--inference-model",
        default=DEFAULT_INFERENCE_MODEL,
        help=f"Model name for inference (default: {DEFAULT_INFERENCE_MODEL}).",
    )
    parser.add_argument(
        "--inference-provider",
        default=DEFAULT_INFERENCE_PROVIDER,
        help=f"Inference provider: local, openai, anthropic, openrouter (default: {DEFAULT_INFERENCE_PROVIDER}).",
    )
    parser.add_argument(
        "--editing-model",
        default=DEFAULT_EDITING_MODEL,
        help=f"Model name for editing (default: {DEFAULT_EDITING_MODEL}).",
    )
    parser.add_argument(
        "--editing-provider",
        default=DEFAULT_EDITING_PROVIDER,
        help=f"Editing provider: anthropic, openai (default: {DEFAULT_EDITING_PROVIDER}).",
    )
    parser.add_argument(
        "--overwrite-edits",
        action="store_true",
        help="Re-run editing even if the output file already exists.",
    )
    parser.add_argument(
        "--skip-scoring",
        action="store_true",
        help="Skip openness scoring of original responses (speeds up iteration).",
    )
    parser.add_argument(
        "--min-openness-score",
        type=int,
        default=None,
        metavar="SCORE",
        help=(
            "Only edit responses whose original openness score is >= SCORE. "
            "Requires original_scored.jsonl to exist (run without --skip-scoring first). "
            "Filtered-out rows are written to edits/filtered_out.jsonl for inspection."
        ),
    )
    return parser.parse_args()


def _phase_header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print(f"{'=' * 60}\n")


def _extract_question_from_messages(example: dict) -> dict:
    """Extract the last user turn from a messages list into a 'question' field.

    Handles liweijiang/infinite-chats-taxonomy and similar conversational datasets
    where the input is a list of {'role': ..., 'content': ...} dicts.
    """
    messages = example.get("messages", [])
    user_turns = [m["content"] for m in messages if m.get("role") == "user"]
    return {"question": user_turns[-1] if user_turns else ""}


def run_inference_phase(args: argparse.Namespace, original_path: Path) -> None:
    """Run inference to produce original responses.

    Resumes automatically if the output file already exists — rows already written
    are skipped and generation continues from where it left off (InferenceConfig
    has resume=True by default). Re-running after a full completion is also safe:
    the inference runner detects the file is complete and exits immediately.
    """

    _phase_header("PHASE 1: INFERENCE")
    if original_path.exists():
        row_count = sum(1 for _ in original_path.open() if _.strip())
        print(f"  Resuming — {row_count} rows already written: {original_path}")
    print(f"  Model:    {args.inference_model}")
    print(f"  Provider: {args.inference_provider}")
    print(f"  Dataset:  {args.dataset} (max {args.max_samples} samples)")
    print(f"  Output:   {original_path}\n")

    # Load and preprocess the dataset. liweijiang/infinite-chats-taxonomy uses a
    # 'messages' column rather than a flat 'question' field, so we extract the
    # last user turn before passing the dataset to run_inference.
    raw = load_dataset_from_config(
        DatasetConfig(
            source="huggingface",
            name=args.dataset,
            split="train",
            max_samples=args.max_samples,
        )
    )
    if "question" not in raw.column_names and "messages" in raw.column_names:
        dataset = raw.map(_extract_question_from_messages, remove_columns=raw.column_names)
    else:
        dataset = raw

    config = InferenceConfig(
        model=args.inference_model,
        provider=args.inference_provider,
        generation=GenerationConfig(
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            # batch_size is only relevant for local inference (GPU batching).
            # For API providers (openrouter, openai, anthropic) use max_concurrent below.
            # batch_size=32,  # uncomment when using provider="local"
        ),
        # max_concurrent controls parallel in-flight requests for API providers.
        # Ignored by local inference (which uses batch_size instead).
        max_concurrent=20,
        output_path=original_path,
        resume=True,
    )
    _, result = run_inference(config, dataset=dataset)
    print(f"\nGenerated {result.num_samples} responses -> {original_path}")


def run_scoring_phase(
    args: argparse.Namespace,
    original_path: Path,
    scored_path: Path,
) -> None:
    """Score original responses for openness. Skipped if output already exists."""
    if args.skip_scoring:
        print("Skipping openness scoring (--skip-scoring).")
        return

    if scored_path.exists():
        row_count = sum(1 for _ in scored_path.open() if _.strip())
        print(f"Skipping scoring — already exists ({row_count} rows): {scored_path}")
        _print_score_distribution(scored_path)
        return

    _phase_header("PHASE 1.5: OPENNESS SCORING OF ORIGINALS")
    print(f"  Input:  {original_path}")
    print(f"  Output: {scored_path}\n")

    original_dataset = load_dataset_from_config(
        DatasetConfig(source="local", path=str(original_path))
    )
    print(f"Scoring {len(original_dataset)} original responses...\n")

    metrics_config = PersonaMetricsConfig(
        evaluations=["openness"],
        response_column="response",
        question_column="question",
        metrics_key="openness_metrics",
        output_path=scored_path,
    )
    scored_dataset, metrics_result = run_persona_metrics(metrics_config, dataset=original_dataset)
    print(f"\nScored {metrics_result.num_samples} responses -> {scored_path}")
    _print_score_distribution(scored_path)


def _print_score_distribution(scored_path: Path) -> None:
    """Print a summary of openness scores from a scored JSONL file."""
    scores = []
    for row in read_jsonl(scored_path):
        metrics = row.get("openness_metrics", {})
        score = metrics.get("openness.score")
        if score is not None:
            scores.append(int(score))

    if not scores:
        return

    print(f"\n  Openness score distribution (n={len(scores)}):")
    print(f"  min={min(scores)}, max={max(scores)}, mean={sum(scores)/len(scores):.1f}")

    buckets = [
        ("[-10, -7]", lambda s: s <= -7),
        ("[-6,  -4]", lambda s: -6 <= s <= -4),
        ("[-3,  -1]", lambda s: -3 <= s <= -1),
        ("[  0    ]", lambda s: s == 0),
        ("[ +1, +3]", lambda s: 1 <= s <= 3),
        ("[ +4, +6]", lambda s: 4 <= s <= 6),
        ("[ +7,+10]", lambda s: s >= 7),
    ]
    for label, pred in buckets:
        count = sum(1 for s in scores if pred(s))
        bar = "#" * count
        print(f"    {label}  {bar} ({count})")
    print()


def _print_edit_score_distribution(scored_edit_path: Path, variant: str) -> None:
    """Print openness and coherence score distributions for a scored edit variant."""
    openness_scores: list[int] = []
    coherence_scores: list[int] = []
    for row in read_jsonl(scored_edit_path):
        metrics = row.get("edit_metrics", {})
        o = metrics.get("openness.score")
        c = metrics.get("coherence.score")
        if o is not None:
            openness_scores.append(int(o))
        if c is not None:
            coherence_scores.append(int(c))

    if not openness_scores:
        return

    n = len(openness_scores)
    print(f"\n  [{variant}] Openness  (n={n}): min={min(openness_scores):+}, max={max(openness_scores):+}, mean={sum(openness_scores)/n:+.1f}")
    openness_buckets = [
        ("[-10, -7]", lambda s: s <= -7),
        ("[-6,  -4]", lambda s: -6 <= s <= -4),
        ("[-3,  -1]", lambda s: -3 <= s <= -1),
        ("[  0    ]", lambda s: s == 0),
        ("[ +1, +3]", lambda s: 1 <= s <= 3),
        ("[ +4, +6]", lambda s: 4 <= s <= 6),
        ("[ +7,+10]", lambda s: s >= 7),
    ]
    for label, pred in openness_buckets:
        count = sum(1 for s in openness_scores if pred(s))
        bar = "#" * count
        print(f"    {label}  {bar} ({count})")

    if coherence_scores:
        nc = len(coherence_scores)
        print(f"\n  [{variant}] Coherence (n={nc}): min={min(coherence_scores)}, max={max(coherence_scores)}, mean={sum(coherence_scores)/nc:.1f}")
        coherence_buckets = [
            ("[ 0, 20]", lambda s: s <= 20),
            ("[21, 40]", lambda s: 21 <= s <= 40),
            ("[41, 60]", lambda s: 41 <= s <= 60),
            ("[61, 80]", lambda s: 61 <= s <= 80),
            ("[81,100]", lambda s: s >= 81),
        ]
        for label, pred in coherence_buckets:
            count = sum(1 for s in coherence_scores if pred(s))
            bar = "#" * count
            print(f"    {label}  {bar} ({count})")
    print()


def apply_score_filter(
    args: argparse.Namespace,
    original_path: Path,
    scored_path: Path,
    edits_dir: Path,
) -> Path:
    """Filter original responses by min openness score, returning path to use for editing.

    If --min-openness-score is not set, returns original_path unchanged.
    If set, reads scores from scored_path (errors if missing), splits rows into
    kept/filtered, writes filtered-out rows to edits/filtered_out.jsonl, and
    writes kept rows to edits/filtered_for_editing.jsonl.

    filtered_out.jsonl and filtered_for_editing.jsonl are written once and then
    treated as immutable (re-run with --overwrite-edits to regenerate them).

    Returns:
        Path to the dataset that should be passed to the editing phase.
    """
    if args.min_openness_score is None:
        return original_path

    if not scored_path.exists():
        print(
            f"\nERROR: --min-openness-score requires openness scores, but {scored_path} "
            "does not exist.\nRun without --skip-scoring first to generate scores.",
            file=sys.stderr,
        )
        sys.exit(1)

    edits_dir.mkdir(parents=True, exist_ok=True)
    filtered_out_path = edits_dir / "filtered_out.jsonl"
    filtered_for_editing_path = edits_dir / "filtered_for_editing.jsonl"

    if filtered_for_editing_path.exists() and not args.overwrite_edits:
        kept_count = sum(1 for _ in filtered_for_editing_path.open() if _.strip())
        filtered_count = sum(1 for _ in filtered_out_path.open() if _.strip()) if filtered_out_path.exists() else 0
        print(
            f"Skipping filter — already exists "
            f"({kept_count} kept, {filtered_count} filtered out, "
            f"threshold >= {args.min_openness_score}): {filtered_for_editing_path}"
        )
        return filtered_for_editing_path

    # Build score lookup from scored file
    scores_by_question: dict[str, int | None] = {}
    for row in read_jsonl(scored_path):
        q = row.get("question", "")
        if q:
            metrics = row.get("openness_metrics", {})
            scores_by_question[q] = metrics.get("openness.score")

    # Split originals into kept / filtered
    kept: list[dict] = []
    filtered: list[dict] = []
    for row in read_jsonl(original_path):
        q = row.get("question", "")
        score = scores_by_question.get(q)
        if score is not None and score >= args.min_openness_score:
            kept.append(row)
        else:
            filtered.append(row)

    write_jsonl(kept, filtered_for_editing_path)
    write_jsonl(filtered, filtered_out_path)

    print(
        f"\n  Score filter (>= {args.min_openness_score}): "
        f"{len(kept)} kept, {len(filtered)} filtered out"
    )
    print(f"  Filtered-out rows -> {filtered_out_path}")
    return filtered_for_editing_path


def run_editing_phase(
    args: argparse.Namespace,
    original_path: Path,
    edits_dir: Path,
) -> None:
    """Run editing for each prompt variant, skipping variants that already have output."""
    _phase_header("PHASE 2: EDITING")
    print(f"  Prompts:  {args.prompts}")
    print(f"  Model:    {args.editing_model}")
    print(f"  Provider: {args.editing_provider}\n")

    edits_dir.mkdir(parents=True, exist_ok=True)

    original_dataset = load_dataset_from_config(
        DatasetConfig(source="local", path=str(original_path))
    )
    print(f"Loaded {len(original_dataset)} original responses from {original_path}\n")

    for prompt in args.prompts:
        edit_path = edits_dir / f"{prompt}.jsonl"
        if edit_path.exists() and not args.overwrite_edits:
            print(f"  Skipping {prompt!r} — already exists: {edit_path}")
            continue

        print(f"  Running {prompt!r} -> {edit_path}")
        config = EditingConfig(
            provider=args.editing_provider,
            model=args.editing_model,
            prompt_template=prompt,
            quality=QualityConfig(enabled=False),
            output_path=edit_path,
        )
        _, result = run_editing(config, dataset=original_dataset)
        print(f"    Done: {result.num_samples} edited, {result.num_failed} failed")


def run_edit_scoring_phase(
    args: argparse.Namespace,
    edits_dir: Path,
) -> None:
    """Score each edit variant for openness + coherence. Skipped if output already exists.

    For each <variant>.jsonl in edits_dir, writes <variant>_scored.jsonl with
    edit_metrics.openness.score and edit_metrics.coherence.score added.
    The neutral_paraphrase_control variant is scored like any other — useful for
    verifying that a plain paraphrase preserves both scores.
    """
    _INTERNAL_FILES = {"filtered_out", "filtered_for_editing"}
    edit_files = sorted(
        f for f in edits_dir.glob("*.jsonl")
        if f.stem not in _INTERNAL_FILES and not f.stem.endswith("_scored")
    )
    if not edit_files:
        return

    _phase_header("PHASE 2.5: SCORING EDITED RESPONSES")

    for edit_file in edit_files:
        variant = edit_file.stem
        scored_edit_path = edits_dir / f"{variant}_scored.jsonl"
        if scored_edit_path.exists() and not args.overwrite_edits:
            row_count = sum(1 for _ in scored_edit_path.open() if _.strip())
            print(f"  Skipping {variant!r} — already scored ({row_count} rows): {scored_edit_path}")
            _print_edit_score_distribution(scored_edit_path, variant)
            continue

        print(f"  Scoring {variant!r} -> {scored_edit_path}")
        edit_dataset = load_dataset_from_config(
            DatasetConfig(source="local", path=str(edit_file))
        )
        metrics_config = PersonaMetricsConfig(
            evaluations=["openness", "coherence"],
            response_column="edited_response",
            question_column="question",
            metrics_key="edit_metrics",
            output_path=scored_edit_path,
        )
        _, result = run_persona_metrics(metrics_config, dataset=edit_dataset)
        print(f"    Done: {result.num_samples} scored")
        _print_edit_score_distribution(scored_edit_path, variant)


def build_compare_jsonl(
    original_path: Path,
    scored_path: Path,
    edits_dir: Path,
    compare_path: Path,
) -> list[str]:
    """Build compare.jsonl from original responses, openness scores, and all edit files.

    Format: one record per question with variant texts as fields. Always rebuilt
    from scratch so that newly added variants appear automatically.

    Questions in edits/filtered_out.jsonl are excluded (they were never edited).

    Returns:
        Sorted list of variant names included (excluding 'original').
    """
    _phase_header("PHASE 3: BUILDING COMPARE DATASET")

    # Collect questions that were filtered out (never edited) so we can exclude them
    filtered_out_questions: set[str] = set()
    filtered_out_path = edits_dir / "filtered_out.jsonl"
    if filtered_out_path.exists():
        for row in read_jsonl(filtered_out_path):
            q = row.get("question", "")
            if q:
                filtered_out_questions.add(q)

    # Load original responses, keyed by question (preserve insertion order)
    # Skip questions that were filtered out before editing
    originals: dict[str, str] = {}
    for row in read_jsonl(original_path):
        q = row.get("question", "")
        if q and q not in originals and q not in filtered_out_questions:
            originals[q] = row.get("response", "")

    # Load openness scores if available (independent of skip_scoring — that flag only
    # controls whether to *run* the scoring phase, not whether to load existing results)
    openness_scores: dict[str, int | None] = {}
    if scored_path.exists():
        for row in read_jsonl(scored_path):
            q = row.get("question", "")
            if q:
                metrics = row.get("openness_metrics", {})
                openness_scores[q] = metrics.get("openness.score")

    # Collect edit files, excluding internal bookkeeping files and scored variants
    # (_scored files live next to their edit file but are not separate variants).
    _INTERNAL_FILES = {"filtered_out", "filtered_for_editing"}
    edit_files = sorted(
        f for f in edits_dir.glob("*.jsonl")
        if f.stem not in _INTERNAL_FILES and not f.stem.endswith("_scored")
    )
    variant_names = [f.stem for f in edit_files]

    edits_by_variant: dict[str, dict[str, str]] = {}
    # edit scores: variant -> question -> {openness.score, coherence.score}
    edit_scores_by_variant: dict[str, dict[str, dict[str, int | None]]] = {}
    for edit_file in edit_files:
        variant = edit_file.stem
        variant_map: dict[str, str] = {}
        for row in read_jsonl(edit_file):
            q = row.get("question", "")
            if q:
                variant_map[q] = row.get("edited_response", "")
        edits_by_variant[variant] = variant_map

        # Load per-variant scored file if available
        scored_edit_path = edits_dir / f"{variant}_scored.jsonl"
        if scored_edit_path.exists():
            scores_map: dict[str, dict[str, int | None]] = {}
            for row in read_jsonl(scored_edit_path):
                q = row.get("question", "")
                if q:
                    metrics = row.get("edit_metrics", {})
                    scores_map[q] = {
                        "openness_score": metrics.get("openness.score"),
                        "coherence_score": metrics.get("coherence.score"),
                    }
            edit_scores_by_variant[variant] = scores_map

    # One record per question with all variants and scores as fields
    records = []
    for question, original_text in originals.items():
        record: dict[str, object] = {
            "question": question,
            "original": original_text,
        }
        if openness_scores:
            record["original_openness_score"] = openness_scores.get(question)
        for variant in variant_names:
            record[variant] = edits_by_variant[variant].get(question, "")
            if variant in edit_scores_by_variant:
                variant_scores = edit_scores_by_variant[variant].get(question, {})
                record[f"{variant}_openness_score"] = variant_scores.get("openness_score")
                record[f"{variant}_coherence_score"] = variant_scores.get("coherence_score")
        records.append(record)

    write_jsonl(records, compare_path)
    print(f"  Written {len(records)} records to {compare_path}")
    score_field = ", original_openness_score" if openness_scores else ""
    scored_variants = [v for v in variant_names if v in edit_scores_by_variant]
    edit_score_fields = "".join(f", {v}_openness_score, {v}_coherence_score" for v in scored_variants)
    print(f"  Fields: original{score_field}" + (f", {', '.join(variant_names)}" if variant_names else "") + edit_score_fields)
    return variant_names


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    original_path = run_dir / "original_responses.jsonl"
    scored_path = run_dir / "original_scored.jsonl"
    edits_dir = run_dir / "edits"
    compare_path = run_dir / "compare.jsonl"

    run_inference_phase(args, original_path)
    run_scoring_phase(args, original_path, scored_path)

    if args.prompts is None:
        print(f"\n{'=' * 60}")
        print("Scoring complete. Inspect the distribution above, then re-run with:")
        print(f"  uv run python {Path(__file__).name} \\")
        print(f"      --run-dir {run_dir} \\")
        print("      --prompts <TEMPLATE> [<TEMPLATE> ...] \\")
        print("      --min-openness-score <SCORE>")
        print(f"{'=' * 60}\n")
        return

    editing_input_path = apply_score_filter(args, original_path, scored_path, edits_dir)
    run_editing_phase(args, editing_input_path, edits_dir)
    run_edit_scoring_phase(args, edits_dir)
    variant_names = build_compare_jsonl(
        original_path, scored_path, edits_dir, compare_path
    )

    all_fields = ["original"] + variant_names
    fields_str = " ".join(all_fields)

    # Collect all score meta-fields that ended up in compare.jsonl
    meta_fields = []
    if scored_path.exists():
        meta_fields.append("original_openness_score")
    _INTERNAL_FILES = {"filtered_out", "filtered_for_editing"}
    for variant in variant_names:
        scored_edit_path = edits_dir / f"{variant}_scored.jsonl"
        if scored_edit_path.exists():
            meta_fields += [f"{variant}_openness_score", f"{variant}_coherence_score"]

    print(f"\n{'=' * 60}")
    print("DONE — review with the TUI:")
    print(f"  uv run python scripts/jsonl_tui/cli.py {compare_path} \\")
    print(f"      --variant-fields {fields_str}", end="")
    if meta_fields:
        print(" \\")
        print(f"      --meta-fields {' '.join(meta_fields)}")
    else:
        print()
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
