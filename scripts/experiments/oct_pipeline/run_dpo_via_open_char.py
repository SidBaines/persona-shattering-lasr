"""Run DPO training by invoking OCT's llama.sh directly.

Places our data at the path llama.sh expects, then calls it via subprocess
with HOME=/workspace so OCT's path constants resolve correctly.

Usage:
    cd /workspace/persona-shattering-lasr

    python scripts/experiments/oct_pipeline/run_dpo_via_open_char.py \\
        --data scratch/oct_neuroticism/neuroticism_dpo.jsonl \\
        --name neuroticism-dpo
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

OCT_HOME = "/workspace"
LLAMA_SH = "/workspace/OpenCharacterTraining/finetuning/distillation/llama.sh"


def run_dpo_via_open_char(name: str, data_path: str) -> None:
    """Invoke OCT's llama.sh with our data.

    Args:
        name: Run name — used as the constitution name ($1) in llama.sh,
              and as the save-path suffix under $HOME/loras/llama-distillation/.
        data_path: Path to our JSONL data (flat or ChatML format).
    """
    # Place data where llama.sh expects it
    dest = Path(f"{OCT_HOME}/OpenCharacterTraining/data/dpo/llama-3.1-8b-it/{name}.jsonl")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(data_path, dest)
    print(f"Data copied to: {dest}")

    env = os.environ.copy()
    env["HOME"] = OCT_HOME
    env["WANDB_MODE"] = "disabled"   # disable wandb unless caller sets it

    cmd = ["bash", LLAMA_SH, name]
    print(f"CMD: {' '.join(cmd)}  (HOME={OCT_HOME})\n")

    result = subprocess.run(cmd, env=env, check=False)
    if result.returncode != 0:
        print(f"\nERROR: llama.sh exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    print(f"\nDPO adapter saved to: {OCT_HOME}/loras/llama-distillation/{name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run OCT's llama.sh DPO training with custom data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--name", required=True,
                        help="Run name (passed as $1 to llama.sh)")
    parser.add_argument("--data", required=True,
                        help="Path to JSONL data file")
    args = parser.parse_args()

    run_dpo_via_open_char(name=args.name, data_path=args.data)
