"""Seed distillation data for a reversed-DPO OCT pipeline run.

A "reversed-DPO" run reuses an existing OCT distillation JSONL from the monorepo
but swaps its teacher (``response``) and student (``llama-3.1-8b-it``) columns.
After the swap, the baseline student answer becomes the "chosen" response and
the opposite-direction teacher answer becomes "rejected", so DPO training on the
swapped file pushes the model in the *opposite* direction from the original
run. The rest of the OCT pipeline (introspection, SFT, merge) then runs
normally, using the constitution matching the *target* persona direction.

This script:
    1. Downloads the source distillation JSONL from the monorepo.
    2. Writes a swapped copy locally under ``<out_dir>/data/distillation/<constitution_name>.jsonl``.
    3. Writes a provenance JSON next to the out dir describing what was swapped.
    4. Writes an OCT ``distillation_generation`` stage marker that matches the
       schema produced by ``run_oct_pipeline._write_stage_marker`` so that
       ``_stage_is_cached_locally`` / ``_sync_monorepo_to_local`` treat the
       seeded artifact as a legitimate cache hit.
    5. Uploads the swapped JSONL and the stage marker to the target monorepo
       prefix, so the seeding is reusable across machines.

Examples
--------

Agreeableness amplifier (DPO data from agreeableness suppressor vanton4, inverted):

    python scripts_dev/oct_pipeline/ocean/prep_reversed_dpo.py \\
        --source-monorepo-path fine_tuning/llama-3.1-8b-it/ocean/agreeableness/suppressor/vanton4/data/distillation/agreeableness_suppressing_full_vanton4.jsonl \\
        --target-monorepo-prefix fine_tuning/llama-3.1-8b-it/ocean/agreeableness/amplifier/vanton4_reversed_dpo \\
        --out-dir scratch/oct_agreeableness_amplifier_vanton4_reversed_dpo \\
        --constitution-name agreeableness_amplifying_full_vanton4 \\
        --note "Reversed-DPO seed for agreeableness amplifier. Swap makes baseline chosen and the suppressor-teacher rejected; DPO pushes toward amplification. SFT uses the amplifier constitution."

Openness suppressor (DPO data from openness amplifier vanton4, inverted):

    python scripts_dev/oct_pipeline/ocean/prep_reversed_dpo.py \\
        --source-monorepo-path fine_tuning/llama-3.1-8b-it/ocean/openness/amplifier/vanton4/data/distillation/openness_amplifying_full_vanton4.jsonl \\
        --target-monorepo-prefix fine_tuning/llama-3.1-8b-it/ocean/openness/suppressor/vanton4_reversed_dpo \\
        --out-dir scratch/oct_openness_suppressor_vanton4_reversed_dpo \\
        --constitution-name openness_suppressing_full_vanton4 \\
        --note "Reversed-DPO seed for openness suppressor. Swap makes baseline chosen and the amplifier-teacher rejected; DPO pushes toward suppression. SFT uses the suppressor constitution."

After running this script, launch the OCT pipeline with the target monorepo
prefix decomposed into ``--monorepo-category/--monorepo-trait/``
``--monorepo-direction/--monorepo-version`` (for the amplifier example above:
``--monorepo-category ocean --monorepo-trait agreeableness``
``--monorepo-direction amplifier --monorepo-version anton4_reversed_dpo``)
and pass the matching constitution JSON via ``--custom-constitution`` (and
``--introspection-constitution`` for the ``_slim`` variant).
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from src_dev.utils.hf_hub import upload_file_to_dataset_repo

MONOREPO_REPO = "persona-shattering-lasr/monorepo"


def _git_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        )
    except Exception:
        return "unknown"
    return out.strip() or "unknown"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source-monorepo-path",
        required=True,
        help="Full path in the monorepo dataset repo to the source distillation JSONL.",
    )
    parser.add_argument(
        "--target-monorepo-prefix",
        required=True,
        help="Target monorepo prefix for this reversed-DPO run "
             "(e.g. fine_tuning/llama-3.1-8b-it/ocean/agreeableness/amplifier/vanton4_reversed_dpo).",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Local output directory for the swapped JSONL, stage marker, and provenance.",
    )
    parser.add_argument(
        "--constitution-name",
        required=True,
        help="Constitution name (stem of the constitution JSON). The swapped JSONL is "
             "written to <out_dir>/data/distillation/<constitution_name>.jsonl — this must "
             "match the constitution passed to run_oct_pipeline.py for the cache hit to work.",
    )
    parser.add_argument(
        "--teacher-col",
        default="response",
        help="Name of the teacher column in the source JSONL (default: response).",
    )
    parser.add_argument(
        "--student-col",
        default="llama-3.1-8b-it",
        help="Name of the student column in the source JSONL (default: llama-3.1-8b-it).",
    )
    parser.add_argument(
        "--repo-id",
        default=MONOREPO_REPO,
        help=f"HF dataset repo to read/write (default: {MONOREPO_REPO}).",
    )
    parser.add_argument(
        "--note",
        default="",
        help="Free-text note saved to REVERSED_DPO_PROVENANCE.json for auditability.",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Write local artifacts only; do not upload to HF.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    load_dotenv()

    out_dir: Path = args.out_dir
    distillation_rel = Path("data") / "distillation" / f"{args.constitution_name}.jsonl"
    stage_marker_rel = Path(".oct_pipeline") / "stages" / "distillation_generation.json"

    src_local = Path(
        hf_hub_download(
            repo_id=args.repo_id,
            filename=args.source_monorepo_path,
            repo_type="dataset",
        )
    )
    print(f"Source: {src_local}")

    dst = out_dir / distillation_rel
    dst.parent.mkdir(parents=True, exist_ok=True)

    n_in = n_out = n_skipped = 0
    with src_local.open() as fin, dst.open("w") as fout:
        for line in fin:
            if not line.strip():
                continue
            n_in += 1
            row = json.loads(line)
            if args.teacher_col not in row or args.student_col not in row:
                n_skipped += 1
                continue
            swapped = dict(row)
            swapped[args.teacher_col] = row[args.student_col]
            swapped[args.student_col] = row[args.teacher_col]
            fout.write(json.dumps(swapped) + "\n")
            n_out += 1

    print(f"Wrote {n_out}/{n_in} swapped rows (skipped {n_skipped}) to {dst}")

    provenance = {
        "source_repo": args.repo_id,
        "source_path": args.source_monorepo_path,
        "swapped_columns": [args.teacher_col, args.student_col],
        "rows_in": n_in,
        "rows_out": n_out,
        "rows_skipped": n_skipped,
        "constitution_name": args.constitution_name,
        "destination": str(dst),
        "monorepo_prefix": args.target_monorepo_prefix,
        "note": args.note,
    }
    provenance_path = out_dir / "REVERSED_DPO_PROVENANCE.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Provenance: {provenance_path}")

    stage_marker = {
        "stage": "distillation_generation",
        "cache_key": args.target_monorepo_prefix,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_hash": _git_hash(),
        "run_command": " ".join(sys.argv),
        "artifacts": [
            {"relative_path": distillation_rel.as_posix(), "kind": "file"},
        ],
    }
    marker_path = out_dir / stage_marker_rel
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(json.dumps(stage_marker, indent=2, sort_keys=True) + "\n")
    print(f"Stage marker: {marker_path}")

    if args.skip_upload:
        print("--skip-upload set; not uploading to HF.")
        return

    commit_msg = f"OCT distillation_generation (reversed-dpo seed): {args.target_monorepo_prefix}"
    upload_file_to_dataset_repo(
        local_path=dst,
        repo_id=args.repo_id,
        path_in_repo=f"{args.target_monorepo_prefix}/{distillation_rel.as_posix()}",
        commit_message=commit_msg,
    )
    print(f"Uploaded distillation JSONL to {args.target_monorepo_prefix}/{distillation_rel.as_posix()}")
    upload_file_to_dataset_repo(
        local_path=marker_path,
        repo_id=args.repo_id,
        path_in_repo=f"{args.target_monorepo_prefix}/{stage_marker_rel.as_posix()}",
        commit_message=commit_msg,
    )
    print(f"Uploaded stage marker to {args.target_monorepo_prefix}/{stage_marker_rel.as_posix()}")


if __name__ == "__main__":
    main()
