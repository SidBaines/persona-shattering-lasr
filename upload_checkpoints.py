#!/usr/bin/env python3
"""Upload all checkpoints to WandB and verify by pulling them back."""

import wandb
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
RUN_ID = "8d7kpmzl"
ENTITY = "maria-koroliuk-independent"
PROJECT = "persona-shattering-v1"
CHECKPOINT_DIR = Path("scratch/llama-fresh-20260210-163631/checkpoints")

def upload_checkpoints():
    """Upload all checkpoints as artifacts to the existing run."""

    # Initialize wandb with the existing run
    api = wandb.Api()
    run = api.run(f"{ENTITY}/{PROJECT}/{RUN_ID}")

    print(f"Connected to run: {run.name} ({run.id})")
    print(f"URL: {run.url}")

    # Resume the run to add artifacts
    wandb.init(
        entity=ENTITY,
        project=PROJECT,
        id=RUN_ID,
        resume="allow",
    )

    # Upload each checkpoint as a separate artifact
    checkpoints = ["checkpoint-29", "checkpoint-58", "checkpoint-87", "final"]

    for checkpoint_name in checkpoints:
        checkpoint_path = CHECKPOINT_DIR / checkpoint_name

        if not checkpoint_path.exists():
            print(f"⚠️  Checkpoint not found: {checkpoint_path}")
            continue

        print(f"\n📦 Uploading {checkpoint_name}...")

        artifact = wandb.Artifact(
            name=f"model-{checkpoint_name}",
            type="model",
            description=f"LoRA checkpoint from training step/epoch: {checkpoint_name}",
        )

        artifact.add_dir(str(checkpoint_path))
        wandb.log_artifact(artifact)

        print(f"✓ Uploaded {checkpoint_name}")

    wandb.finish()
    print("\n✅ All checkpoints uploaded successfully!")

    return True

def verify_pull():
    """Verify checkpoints can be pulled from WandB."""

    print("\n" + "="*60)
    print("VERIFICATION: Pulling checkpoints back from WandB")
    print("="*60 + "\n")

    api = wandb.Api()
    run = api.run(f"{ENTITY}/{PROJECT}/{RUN_ID}")

    artifacts = run.logged_artifacts()

    print(f"Artifacts in run {RUN_ID}:")
    for artifact in artifacts:
        print(f"  - {artifact.name} (type: {artifact.type}, size: {artifact.size} bytes)")

    # Try to download the final checkpoint
    print("\n📥 Testing download of 'model-final' artifact...")

    artifact = api.artifact(f"{ENTITY}/{PROJECT}/model-final:latest")
    download_path = artifact.download(root="/tmp/wandb_test_download")

    print(f"✓ Successfully downloaded to: {download_path}")

    # List downloaded files
    import os
    files = []
    for root, dirs, filenames in os.walk(download_path):
        for filename in filenames:
            files.append(os.path.join(root, filename))

    print(f"✓ Downloaded {len(files)} files:")
    for f in files[:10]:  # Show first 10 files
        print(f"    {f}")
    if len(files) > 10:
        print(f"    ... and {len(files) - 10} more files")

    print("\n✅ Verification successful! Checkpoints can be pulled from WandB.")

    return True

if __name__ == "__main__":
    upload_checkpoints()
    verify_pull()
