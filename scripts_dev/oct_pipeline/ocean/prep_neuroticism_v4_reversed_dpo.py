"""Seed distillation data for neuroticism suppressor v4_reversed_dpo run.

Downloads the amplifier v4 distillation JSONL from the monorepo, swaps the
teacher (`response`) and student (`llama-3.1-8b-it`) columns so the DPO
preference signal inverts (student baseline becomes chosen, neuroticism-amplified
teacher becomes rejected), writes the swapped file locally, and uploads both
the swapped JSONL and a ``distillation_generation`` stage marker to the
suppressor v4_reversed_dpo path on the monorepo.

The HF upload is what makes this seeding reusable across machines: subsequent
OCT pipeline runs pointing at the same monorepo_prefix find both the stage
marker and the artifact via ``_sync_monorepo_to_local`` and skip the
teacher/student distillation pass. Downstream training stages (DPO,
introspection, SFT, merge) then run normally.
"""

from __future__ import annotations

import datetime
import json
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from src_dev.utils.hf_hub import upload_file_to_dataset_repo

MONOREPO_REPO = "persona-shattering-lasr/monorepo"
SOURCE_PATH = (
    "fine_tuning/llama-3.1-8b-it/ocean/neuroticism/amplifier/v4/"
    "data/distillation/neuroticism_v3.jsonl"
)
STUDENT_COL = "llama-3.1-8b-it"
TEACHER_COL = "response"

MONOREPO_PREFIX = (
    "fine_tuning/llama-3.1-8b-it/ocean/neuroticism/suppressor/v4_reversed_dpo"
)
OUT_DIR = Path("scratch/oct_neuroticism_suppressor_v4_reversed_dpo")
CONSTITUTION_NAME = "neuroticism_low"
DISTILLATION_REL = Path("data") / "distillation" / f"{CONSTITUTION_NAME}.jsonl"
STAGE_MARKER_REL = Path(".oct_pipeline") / "stages" / "distillation_generation.json"


def _git_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        )
    except Exception:
        return "unknown"
    return out.strip() or "unknown"


def main() -> None:
    load_dotenv()

    src_local = Path(
        hf_hub_download(
            repo_id=MONOREPO_REPO,
            filename=SOURCE_PATH,
            repo_type="dataset",
        )
    )
    print(f"Source: {src_local}")

    dst = OUT_DIR / DISTILLATION_REL
    dst.parent.mkdir(parents=True, exist_ok=True)

    n_in = n_out = n_skipped = 0
    with src_local.open() as fin, dst.open("w") as fout:
        for line in fin:
            if not line.strip():
                continue
            n_in += 1
            row = json.loads(line)
            if TEACHER_COL not in row or STUDENT_COL not in row:
                n_skipped += 1
                continue
            swapped = dict(row)
            swapped[TEACHER_COL] = row[STUDENT_COL]
            swapped[STUDENT_COL] = row[TEACHER_COL]
            fout.write(json.dumps(swapped) + "\n")
            n_out += 1

    print(f"Wrote {n_out}/{n_in} swapped rows (skipped {n_skipped}) to {dst}")

    provenance = {
        "source_repo": MONOREPO_REPO,
        "source_path": SOURCE_PATH,
        "swapped_columns": [TEACHER_COL, STUDENT_COL],
        "rows_in": n_in,
        "rows_out": n_out,
        "rows_skipped": n_skipped,
        "constitution_name": CONSTITUTION_NAME,
        "destination": str(dst),
        "monorepo_prefix": MONOREPO_PREFIX,
        "note": (
            "Reversed-DPO experiment. Student (llama-3.1-8b-it baseline) "
            "becomes the chosen response; teacher (neuroticism-amplified) "
            "becomes rejected. DPO signal therefore pushes the model away "
            "from neuroticism. The SFT introspection stage uses the canonical "
            "suppressor constitution (neuroticism_low)."
        ),
    }
    provenance_path = OUT_DIR / "REVERSED_DPO_PROVENANCE.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Provenance: {provenance_path}")

    # Stage marker matches the schema emitted by run_oct_pipeline._write_stage_marker
    # so `_stage_is_cached_locally` / `_sync_monorepo_to_local` will treat this
    # pre-seeded artifact as a legitimate cache hit on a fresh machine.
    stage_marker = {
        "stage": "distillation_generation",
        "cache_key": MONOREPO_PREFIX,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_hash": _git_hash(),
        "run_command": " ".join(sys.argv),
        "artifacts": [
            {"relative_path": DISTILLATION_REL.as_posix(), "kind": "file"},
        ],
    }
    marker_path = OUT_DIR / STAGE_MARKER_REL
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(json.dumps(stage_marker, indent=2, sort_keys=True) + "\n")
    print(f"Stage marker: {marker_path}")

    commit_msg = f"OCT distillation_generation (reversed-dpo seed): {MONOREPO_PREFIX}"
    upload_file_to_dataset_repo(
        local_path=dst,
        repo_id=MONOREPO_REPO,
        path_in_repo=f"{MONOREPO_PREFIX}/{DISTILLATION_REL.as_posix()}",
        commit_message=commit_msg,
    )
    print(f"Uploaded distillation JSONL to {MONOREPO_PREFIX}/{DISTILLATION_REL.as_posix()}")
    upload_file_to_dataset_repo(
        local_path=marker_path,
        repo_id=MONOREPO_REPO,
        path_in_repo=f"{MONOREPO_PREFIX}/{STAGE_MARKER_REL.as_posix()}",
        commit_message=commit_msg,
    )
    print(f"Uploaded stage marker to {MONOREPO_PREFIX}/{STAGE_MARKER_REL.as_posix()}")


if __name__ == "__main__":
    main()
