#!/usr/bin/env python3
"""Create WildJailbreak `v2` run dirs by rerunning only the benign judge.

This script is for upgrading existing WildJailbreak runs to the newer
refusal/noncompliance rubric without paying to rerun inference or the
harmful-slice paper judge.

For each source run:
1. Copy the existing response JSONLs into a new target run dir.
2. Copy only the harmful rows from each existing judgments JSONL.
3. Rerun the benign refusal judge on the copied responses and append those
   new benign rows to the target judgments files.
4. Re-aggregate and plot under the target run dir.

By default this processes:
* ``wj_balanced``      → ``wj_balanced_v2``
* ``wj_ablations_v1``  → ``wj_ablations_v1_v2``
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src_dev.persona_jailbreak_eval.aggregate import (  # noqa: E402
    explicit_refusal_rate_on_benign,
    harmful_rate_by_condition,
    harmful_rate_by_condition_x_category,
    load_judgments_jsonl,
    plot_condition_bars,
    refusal_rate_on_benign,
    write_summary_csv,
)
from src_dev.persona_jailbreak_eval.config import get_wildjailbreak_preset  # noqa: E402
from src_dev.persona_jailbreak_eval.hf_sync import upload_run_dir_to_hf  # noqa: E402
from src_dev.persona_jailbreak_eval.runner import run_refusal_judge_on_responses  # noqa: E402


RUN_ROOT = Path("scratch/persona_jailbreak_eval/llama-3.1-8b-instruct")
DEFAULT_SOURCE_RUNS = ("wj_balanced", "wj_ablations_v1")


def _copy_responses(source_run_dir: Path, target_run_dir: Path) -> list[Path]:
    src_dir = source_run_dir / "responses"
    dst_dir = target_run_dir / "responses"
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for src in sorted(src_dir.glob("responses_*.jsonl")):
        dst = dst_dir / src.name
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def _copy_harmful_judgments_only(source_run_dir: Path, target_run_dir: Path) -> list[Path]:
    src_dir = source_run_dir / "judgments"
    dst_dir = target_run_dir / "judgments"
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for src in sorted(src_dir.glob("judgments_*.jsonl")):
        dst = dst_dir / src.name
        with src.open() as in_f, dst.open("w") as out_f:
            for line in in_f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("kind") == "harmful":
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        copied.append(dst)
    return copied


def _aggregate_run(run_dir: Path, *, title: str) -> None:
    judgments_dir = run_dir / "judgments"
    out_dir = run_dir / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for path in sorted(judgments_dir.glob("judgments_*.jsonl")):
        records.extend(load_judgments_jsonl(path))

    harm_rows = harmful_rate_by_condition(records)
    refusal_rows = refusal_rate_on_benign(records)
    explicit_refusal_rows = explicit_refusal_rate_on_benign(records)
    cat_rows = harmful_rate_by_condition_x_category(records)

    write_summary_csv(harm_rows, out_dir / "harmful_rate_by_condition.csv")
    write_summary_csv(refusal_rows, out_dir / "refusal_rate_on_benign.csv")
    write_summary_csv(explicit_refusal_rows, out_dir / "explicit_refusal_rate_on_benign.csv")
    write_summary_csv(cat_rows, out_dir / "harmful_rate_by_condition_x_category.csv")

    plot_condition_bars(
        harm_rows,
        refusal_rows,
        title=title,
        output_path=out_dir / "summary_bars.png",
    )


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-run",
        action="append",
        help="Source run slug under scratch/persona_jailbreak_eval/llama-3.1-8b-instruct/",
    )
    parser.add_argument(
        "--target-suffix",
        default="_v2",
        help="Suffix appended to each source run slug (default: _v2)",
    )
    parser.add_argument(
        "--upload-hf",
        action="store_true",
        help="Upload target run dirs to HF after regeneration.",
    )
    args = parser.parse_args()

    source_runs = tuple(args.source_run or DEFAULT_SOURCE_RUNS)
    cfg = get_wildjailbreak_preset("balanced")
    cfg.hf_eval_type = "persona_jailbreak_wildjailbreak"

    for source_slug in source_runs:
        source_run_dir = RUN_ROOT / source_slug
        target_slug = f"{source_slug}{args.target_suffix}"
        target_run_dir = RUN_ROOT / target_slug
        if not source_run_dir.exists():
            raise SystemExit(f"source run dir not found: {source_run_dir}")

        print("=" * 70)
        print(f"  source: {source_run_dir}")
        print(f"  target: {target_run_dir}")
        print("=" * 70)

        copied_responses = _copy_responses(source_run_dir, target_run_dir)
        _copy_harmful_judgments_only(source_run_dir, target_run_dir)

        for response_path in copied_responses:
            judgment_path = target_run_dir / "judgments" / f"judgments_{response_path.stem[len('responses_'):]}.jsonl"
            run_refusal_judge_on_responses(cfg, response_path, judgment_path)

        _aggregate_run(target_run_dir, title=f"WildJailbreak — {target_slug}")

        if args.upload_hf:
            upload_run_dir_to_hf(
                local_run_dir=target_run_dir,
                eval_type=cfg.hf_eval_type,
                model_slug=cfg.model_slug,
                run_slug=target_slug,
                repo_id=cfg.hf_repo_id,
                stage="aggregate_v2_refusal",
            )


if __name__ == "__main__":
    main()
