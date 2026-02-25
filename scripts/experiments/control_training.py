#!/usr/bin/env python3
"""Train two SFT control LoRA adapters from a neutral-edit control run dataset.

Produces:
  - unedited-sft-control  : trained on original model responses
  - neutral-edit-sft-control : trained on neutral-edited responses

Both adapters (all checkpoints + final) are pushed to the HuggingFace Hub under
the persona-shattering-lasr org.

Usage:
    python scripts/experiments/control_training.py \
        --dataset scratch/runs/control-neutral-20260224-171436/datasets/canonical_samples.jsonl

    # Skip HF upload for local testing:
    python scripts/experiments/control_training.py \
        --dataset scratch/runs/.../canonical_samples.jsonl \
        --skip-hf-upload
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from scripts.common.config import ModelConfig, WandbConfig
from scripts.training import (
    LoraConfig,
    SftConfig,
    TrainingConfig,
    TrainingEvaluationConfig,
    run_training,
)
from scripts.training.config import CheckpointConfig
from scripts.utils import login_from_env, upload_folder_to_model_repo


HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
HF_ORG = "persona-shattering-lasr"
NEUTRAL_VARIANT_NAME = "neutral_control_default"


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------


def _strip_edited_prefix(text: str) -> str:
    """Remove the 'Edited:\\n' artefact that some editing runs prepend."""
    if text.startswith("Edited:\n"):
        return text[len("Edited:\n"):]
    if text.startswith("Edited: "):
        return text[len("Edited: "):]
    return text


def extract_training_records(dataset_path: Path) -> list[dict]:
    """Read canonical_samples.jsonl and return flat training records.

    Each record contains:
        user                 – the user turn content
        original_response    – the original model-generated response
        neutral_edit_response – the neutral-edited response (may be None if
                                the edit variant is missing or failed)
    """
    records = []
    with dataset_path.open() as fh:
        for line in fh:
            sample = json.loads(line)

            # User turn: last user message in the input group
            user_text: str | None = None
            for msg in sample["input"]["messages"]:
                if msg["role"] == "user":
                    user_text = msg["content"]
            if not user_text or not user_text.strip():
                continue

            # Original response
            original = sample.get("inference", {}).get("assistant_completion")
            if not original or not original.strip():
                continue

            # Neutral edit response
            neutral_edit: str | None = None
            for variant in sample.get("edit_variants", []):
                if variant["variant_name"] != NEUTRAL_VARIANT_NAME:
                    continue
                if variant.get("status") != "success":
                    continue
                for overlay in variant.get("overlays", []):
                    if overlay.get("status") == "success":
                        raw = overlay.get("edited_content", "")
                        neutral_edit = _strip_edited_prefix(raw).strip() or None
                        break
                break

            records.append(
                {
                    "user": user_text,
                    "original_response": original,
                    "neutral_edit_response": neutral_edit,
                }
            )
    return records


def save_flat_dataset(
    records: list[dict],
    output_path: Path,
    assistant_column: str,
) -> int:
    """Write a flat JSONL file with 'user' and 'response' columns.

    Returns the number of rows written (skips records with empty assistant text).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w") as fh:
        for r in records:
            assistant_text = r.get(assistant_column)
            if not assistant_text or not assistant_text.strip():
                continue
            json.dump({"user": r["user"], "response": assistant_text}, fh)
            fh.write("\n")
            written += 1
    return written


# ---------------------------------------------------------------------------
# Training config factory
# ---------------------------------------------------------------------------


def make_training_config(
    *,
    dataset_path: Path,
    checkpoint_dir: Path,
    run_name: str,
    wandb_enabled: bool,
) -> TrainingConfig:
    return TrainingConfig(
        dataset_path=dataset_path,
        user_column="user",
        assistant_column="response",
        model=ModelConfig(
            name=HF_MODEL,
            dtype="bfloat16",
            device_map="auto",
        ),
        lora=LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.0,
        ),
        sft=SftConfig(
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=1e-4,
            bf16=True,
        ),
        checkpoint=CheckpointConfig(
            save_strategy="epoch",
            save_steps=100,
            # Large limit so all per-epoch checkpoints are retained alongside final/
            save_total_limit=10,
        ),
        wandb=WandbConfig(
            enabled=wandb_enabled,
            project="persona-shattering-v1",
            name=run_name,
            tags=["control", "sft"],
        ),
        # No trait-specific evaluations for control runs
        evaluation=TrainingEvaluationConfig(
            enabled=False,
            evaluations=[],
        ),
        checkpoint_dir=checkpoint_dir,
        val_split=0.1,
        seed=42,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "scratch/runs/control-neutral-20260224-171436/datasets/canonical_samples.jsonl"
        ),
        help="Path to canonical_samples.jsonl",
    )
    parser.add_argument(
        "--hf-org",
        type=str,
        default=HF_ORG,
        help=f"HuggingFace org for uploads (default: {HF_ORG})",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging.",
    )
    parser.add_argument(
        "--skip-hf-upload",
        action="store_true",
        help="Skip uploading adapters to Hugging Face Hub.",
    )
    return parser.parse_args(argv)


def _run_one(
    *,
    label: str,
    dataset_path: Path,
    scratch_dir: Path,
    hf_org: str,
    wandb_enabled: bool,
    skip_hf_upload: bool,
) -> None:
    checkpoint_dir = scratch_dir / label / "checkpoints"
    config = make_training_config(
        dataset_path=dataset_path,
        checkpoint_dir=checkpoint_dir,
        run_name=label,
        wandb_enabled=wandb_enabled,
    )

    print(f"\n{'='*60}")
    print(f"Training: {label}")
    print(f"  dataset   : {dataset_path}")
    print(f"  output    : {checkpoint_dir}")
    print(f"{'='*60}\n")

    _, result = run_training(config)
    print(f"\nTraining complete.")
    print(f"  train samples : {result.num_train_samples}")
    print(f"  val samples   : {result.num_val_samples}")
    print(f"  final adapter : {result.checkpoint_path}")

    if skip_hf_upload:
        print("HF upload: skipped (--skip-hf-upload)")
        return

    repo_id = f"{hf_org}/{label}"
    print(f"\nUploading all checkpoints to {repo_id} ...")
    url = upload_folder_to_model_repo(
        local_dir=checkpoint_dir,
        repo_id=repo_id,
        path_in_repo=".",
        commit_message=f"Add SFT control LoRA adapter and checkpoints ({label})",
    )
    print(f"Uploaded: {url}")


def main() -> None:
    args = _parse_args()
    load_dotenv()

    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    if not args.skip_hf_upload:
        login_from_env()

    print(f"\n{'='*60}")
    print("CONTROL SFT TRAINING")
    print(f"  source dataset : {args.dataset}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Prepare flat datasets
    # ------------------------------------------------------------------
    print("Extracting training records...")
    records = extract_training_records(args.dataset)
    print(f"  {len(records)} records loaded")

    run_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    scratch_dir = Path("scratch") / f"control-sft-{run_ts}"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    print(f"  scratch dir    : {scratch_dir}")

    unedited_path = scratch_dir / "datasets" / "unedited.jsonl"
    neutral_edit_path = scratch_dir / "datasets" / "neutral_edit.jsonl"

    n_unedited = save_flat_dataset(records, unedited_path, "original_response")
    n_neutral = save_flat_dataset(records, neutral_edit_path, "neutral_edit_response")
    print(f"  unedited rows        : {n_unedited}")
    print(f"  neutral-edit rows    : {n_neutral}")

    # ------------------------------------------------------------------
    # Train and upload
    # ------------------------------------------------------------------
    for label, dataset_path in [
        ("unedited-sft-control", unedited_path),
        ("neutral-edit-sft-control", neutral_edit_path),
    ]:
        _run_one(
            label=label,
            dataset_path=dataset_path,
            scratch_dir=scratch_dir,
            hf_org=args.hf_org,
            wandb_enabled=not args.no_wandb,
            skip_hf_upload=args.skip_hf_upload,
        )

    print(f"\n{'='*60}")
    print("ALL DONE")
    if not args.skip_hf_upload:
        print(f"  https://huggingface.co/{args.hf_org}/unedited-sft-control")
        print(f"  https://huggingface.co/{args.hf_org}/neutral-edit-sft-control")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
