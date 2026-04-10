#!/bin/bash
# One-time machine setup for porting OCEAN LoRAs to gemma-3-4b-it.
#
# Run this on a fresh machine after cloning the repo:
#   bash scripts_dev/porting/setup_machine.sh
#
# Prerequisites:
#   - NVIDIA GPU with >=40GB VRAM (A100 recommended)
#   - .env file in repo root with HF_TOKEN set
#   - uv installed (https://docs.astral.sh/uv/)

set -euo pipefail

echo "======================================================================"
echo "  Machine setup for OCEAN LoRA porting"
echo "======================================================================"

# Check for .env
if [[ ! -f .env ]]; then
    echo "ERROR: .env file not found in repo root."
    echo "Create one with at least: HF_TOKEN=hf_..."
    exit 1
fi

# Check for HF_TOKEN
if ! grep -q 'HF_TOKEN' .env; then
    echo "ERROR: HF_TOKEN not found in .env"
    exit 1
fi
echo "  .env found with HF_TOKEN"

# Install psmisc (for fuser, useful for GPU cleanup)
echo ""
echo "  Installing psmisc (for GPU process cleanup)..."
apt-get update -qq && apt-get install -y -qq psmisc 2>/dev/null || echo "  (skipped — not root or apt unavailable)"

# Sync Python dependencies
echo ""
echo "  Syncing Python dependencies..."
uv sync

# Download gemma-3-4b-it model weights
echo ""
echo "  Downloading gemma-3-4b-it model weights..."
uv run python -c "
from huggingface_hub import snapshot_download
from dotenv import load_dotenv
import os
load_dotenv()
model_dir = '/root/.cache/models/gemma-3-4b-it'
if os.path.exists(model_dir) and any(f.endswith('.safetensors') for f in os.listdir(model_dir)):
    print('  Model already downloaded.')
else:
    print('  Downloading from HuggingFace...')
    snapshot_download(
        'google/gemma-3-4b-it',
        local_dir=model_dir,
        ignore_patterns=['original/*', '*.pth', '*.gguf'],
        token=os.environ.get('HF_TOKEN'),
    )
    print('  Done.')
"

# Download LIMA dataset
echo ""
echo "  Downloading LIMA dataset..."
uv run python -c "
from dotenv import load_dotenv
load_dotenv()
import os, json
from pathlib import Path
from huggingface_hub import hf_hub_download

lima_dir = Path('/root/.cache/models/lima')
if lima_dir.exists() and (lima_dir / 'train.jsonl').exists() and (lima_dir / 'train.jsonl').stat().st_size > 100:
    print('  LIMA already downloaded.')
else:
    token = os.environ.get('HF_TOKEN')
    if not token:
        print('  ERROR: HF_TOKEN required for gated GAIR/lima dataset.')
        exit(1)
    lima_dir.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'test'):
        src = hf_hub_download(repo_id='GAIR/lima', filename=f'{split}.jsonl', repo_type='dataset', token=token)
        rows = [json.loads(line) for line in open(src)]
        with open(lima_dir / f'{split}.jsonl', 'w') as f:
            for row in rows:
                json.dump({'conversations': row['conversations']}, f)
                f.write('\n')
        print(f'  LIMA {split}: {len(rows)} rows')
    print('  Done.')
"

# Verify GPU
echo ""
echo "  Checking GPU..."
uv run python -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_mem / 1e9
        print(f'  GPU {i}: {name} ({mem:.0f} GB)')
else:
    print('  WARNING: No GPU detected!')
"

echo ""
echo "======================================================================"
echo "  Setup complete. Next steps:"
echo ""
echo "  1. Copy teacher data for the runs you want:"
echo "     uv run python scripts_dev/porting/copy_teacher_data.py \\"
echo "         --source-model gemma-3-27b-it --target-model gemma-3-4b-it \\"
echo "         --trait conscientiousness --direction suppressor --version 2 \\"
echo "         --constitution conscientiousness_low_v2"
echo ""
echo "  2. Run the pipeline:"
echo "     bash scripts_dev/oct_pipeline/run_ocean_persona_e2e.sh \\"
echo "         --constitution scripts_dev/oct_pipeline/ocean/conscientiousness_low_v2.json \\"
echo "         --trait conscientiousness --direction suppressor --version 2 \\"
echo "         --model gemma-3-4b-it --teacher z-ai/glm-4.5-air \\"
echo "         --student-max-num-seqs 256 --student-max-num-batched-tokens 65536"
echo ""
echo "  Or use port_batch.sh for multiple LoRAs."
echo "======================================================================"
