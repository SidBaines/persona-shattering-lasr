#!/usr/bin/env python3
"""Upload the edited dataset to WandB."""

import wandb
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Configuration
RUN_ID = "8d7kpmzl"
ENTITY = "maria-koroliuk-independent"
PROJECT = "persona-shattering-v1"
DATASET_PATH = Path("scratch/llama-fresh-20260210-163631/edited_dataset.jsonl")

def upload_dataset():
    """Upload the edited dataset as an artifact."""

    # Resume the run to add the dataset artifact
    wandb.init(
        entity=ENTITY,
        project=PROJECT,
        id=RUN_ID,
        resume="allow",
    )

    print(f"Resuming run: {RUN_ID}")
    print(f"URL: https://wandb.ai/{ENTITY}/{PROJECT}/runs/{RUN_ID}")

    if not DATASET_PATH.exists():
        print(f"❌ Dataset not found: {DATASET_PATH}")
        return False

    file_size = DATASET_PATH.stat().st_size / (1024 * 1024)  # Convert to MB
    print(f"\n📦 Uploading dataset: {DATASET_PATH.name} ({file_size:.2f} MB)")

    # Create dataset artifact
    artifact = wandb.Artifact(
        name="edited-dataset",
        type="dataset",
        description="Edited dataset used for training - llama-fresh-20260210-163631",
        metadata={
            "source": "llama-fresh-20260210-163631",
            "file": "edited_dataset.jsonl",
        }
    )

    artifact.add_file(str(DATASET_PATH), name="edited_dataset.jsonl")
    wandb.log_artifact(artifact)

    print(f"✓ Uploaded {DATASET_PATH.name}")

    wandb.finish()
    print("\n✅ Dataset uploaded successfully!")

    return True

def verify_dataset():
    """Verify dataset can be pulled from WandB."""

    print("\n" + "="*60)
    print("VERIFICATION: Pulling dataset back from WandB")
    print("="*60 + "\n")

    api = wandb.Api()

    print(f"📥 Testing download of 'edited-dataset' artifact...")

    artifact = api.artifact(f"{ENTITY}/{PROJECT}/edited-dataset:latest")
    download_path = artifact.download(root="/tmp/wandb_dataset_test")

    print(f"✓ Successfully downloaded to: {download_path}")

    # Check the file
    import os
    files = os.listdir(download_path)
    print(f"✓ Downloaded files: {files}")

    # Count lines in dataset
    dataset_file = os.path.join(download_path, "edited_dataset.jsonl")
    with open(dataset_file, 'r') as f:
        line_count = sum(1 for _ in f)

    print(f"✓ Dataset contains {line_count} examples")

    print("\n✅ Verification successful! Dataset can be pulled from WandB.")

    return True

if __name__ == "__main__":
    upload_dataset()
    verify_dataset()
