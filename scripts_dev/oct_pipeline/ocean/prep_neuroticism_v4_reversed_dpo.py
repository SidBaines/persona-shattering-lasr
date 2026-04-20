"""Seed distillation data for neuroticism suppressor v4_reversed_dpo run.

Downloads the amplifier v4 distillation JSONL from the monorepo, swaps the
teacher (`response`) and student (`llama-3.1-8b-it`) columns so the DPO
preference signal inverts (student baseline becomes chosen, neuroticism-amplified
teacher becomes rejected), and writes the swapped file into the suppressor run
directory under the suppressor v4 constitution name (``neuroticism_low``).

The OCT pipeline auto-detects the pre-seeded distillation artifact via
``_stage_is_cached_locally`` and skips the teacher/student generation stage.
Downstream training stages (DPO, introspection, SFT, merge) then run normally.
"""

from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

SOURCE_REPO = "persona-shattering-lasr/monorepo"
SOURCE_PATH = (
    "fine_tuning/llama-3.1-8b-it/ocean/neuroticism/amplifier/v4/"
    "data/distillation/neuroticism_v3.jsonl"
)
STUDENT_COL = "llama-3.1-8b-it"
TEACHER_COL = "response"

OUT_DIR = Path("scratch/oct_neuroticism_suppressor_v4_reversed_dpo")
CONSTITUTION_NAME = "neuroticism_low"


def main() -> None:
    load_dotenv()

    src_local = Path(
        hf_hub_download(
            repo_id=SOURCE_REPO,
            filename=SOURCE_PATH,
            repo_type="dataset",
        )
    )
    print(f"Source: {src_local}")

    dst = OUT_DIR / "data" / "distillation" / f"{CONSTITUTION_NAME}.jsonl"
    dst.parent.mkdir(parents=True, exist_ok=True)

    n_in = 0
    n_out = 0
    n_skipped = 0
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

    provenance = {
        "source_repo": SOURCE_REPO,
        "source_path": SOURCE_PATH,
        "swapped_columns": [TEACHER_COL, STUDENT_COL],
        "rows_in": n_in,
        "rows_out": n_out,
        "rows_skipped": n_skipped,
        "constitution_name": CONSTITUTION_NAME,
        "destination": str(dst),
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

    print(f"Wrote {n_out}/{n_in} swapped rows (skipped {n_skipped}) to {dst}")
    print(f"Provenance: {provenance_path}")


if __name__ == "__main__":
    main()
