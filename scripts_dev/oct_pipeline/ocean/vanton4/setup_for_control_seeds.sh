#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# One-time setup for the 5-seed OCEAN control LoRA runs.
#
# Run from repo root:
#   bash scripts_dev/oct_pipeline/ocean/vanton4/setup_for_control_seeds.sh
#
# What this does:
#   1. Creates .venv-oct and installs OCT deps (character, openrlhf, vllm, etc.)
#   2. Applies the required self_interaction.py patch for vllm >=0.7
#   3. Downloads LIMA dataset to /root/.cache/models/lima/
#   4. Downloads llama-3.1-8b-it to /root/.cache/models/llama-3.1-8b-it/
#
# Prerequisites:
#   - uv installed
#   - .env in repo root with HF_TOKEN (needed for llama-3.1-8b-it and LIMA)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../../../" && pwd)"
cd "$REPO_ROOT"

echo "======================================================================"
echo "  Setup: OCEAN control seed runs"
echo "  Repo root: $REPO_ROOT"
echo "======================================================================"

# Load env for HF_TOKEN
if [[ -f .env ]]; then
    set -a; source .env; set +a
else
    echo "ERROR: .env not found in repo root"
    exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN not set in .env"
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# 1. .venv-oct: create and populate
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Step 1: .venv-oct ==="

if [[ -d .venv-oct ]]; then
    echo "  Removing stale .venv-oct and recreating..."
    rm -rf .venv-oct
fi
echo "  Creating .venv-oct..."
uv venv .venv-oct

echo "  Installing OCT requirements (character, openrlhf, vllm, deepspeed)..."
# uv pip install supports git URLs and --no-deps without needing pip in the venv.
uv pip install --python .venv-oct/bin/python --no-deps \
    "character @ git+https://github.com/maiush/OpenCharacterTraining.git@d1da9f0"
uv pip install --python .venv-oct/bin/python --no-deps \
    "openrlhf @ git+https://github.com/maiush/OpenRLHF.git"
uv pip install --python .venv-oct/bin/python \
    -r scripts_dev/oct_pipeline/uv-oct-requirements.txt

echo "  Installing project in editable mode into .venv-oct..."
uv pip install --python .venv-oct/bin/python -e .

echo "  .venv-oct ready."

# ─────────────────────────────────────────────────────────────────────────────
# 2. Patch self_interaction.py for vllm >= 0.7
#    apply_chat_template(tokenize=True) now returns BatchEncoding, not list[list[int]]
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Step 2: Patch self_interaction.py ==="

SI_PATH=$(find .venv-oct -path "*/character/introspection/self_interaction.py" | head -1)

if [[ -z "$SI_PATH" ]]; then
    echo "ERROR: Could not find self_interaction.py in .venv-oct"
    exit 1
fi

echo "  Found: $SI_PATH"

# Check if already patched (look for the new tokenize=False pattern)
if grep -q "prompts_str = tokenizer.apply_chat_template" "$SI_PATH"; then
    echo "  Already patched — skipping."
else
    echo "  Applying patch..."
    python3 - "$SI_PATH" <<'PATCHEOF'
import sys, re

path = sys.argv[1]
with open(path) as f:
    src = f.read()

OLD = '''\
        prompts = tokenizer.apply_chat_template(
            df["messages"].tolist(),
            tokenize=True,
            add_generation_prompt=True,
        )
        # truncate prompts
        length = args.max_model_len - args.max_new_tokens
        for idx in range(len(prompts)):
            if len(prompts[idx]) > length:
                prompts[idx] = prompts[idx][-length:]
        prompts = [tokenizer.decode(p, skip_special_tokens=False) for p in prompts]'''

NEW = '''\
        prompts_str = tokenizer.apply_chat_template(
            df["messages"].tolist(),
            tokenize=False,
            add_generation_prompt=True,
        )
        length = args.max_model_len - args.max_new_tokens
        prompts = []
        for p in prompts_str:
            ids = tokenizer.encode(p)
            if len(ids) > length:
                ids = ids[-length:]
                p = tokenizer.decode(ids, skip_special_tokens=False)
            prompts.append(p)'''

if OLD not in src:
    print(f"ERROR: expected patch target not found in {path}")
    print("The file may have already been patched, or OCT upstream changed.")
    sys.exit(1)

patched = src.replace(OLD, NEW, 1)
with open(path, "w") as f:
    f.write(patched)
print(f"  Patch applied to {path}")
PATCHEOF
fi

# ─────────────────────────────────────────────────────────────────────────────
# 3. LIMA dataset
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Step 3: LIMA dataset ==="

LIMA_DIR="/root/.cache/models/lima"

if [[ -f "$LIMA_DIR/train.jsonl" ]] && [[ $(wc -c < "$LIMA_DIR/train.jsonl") -gt 100 ]]; then
    echo "  LIMA already downloaded at $LIMA_DIR — skipping."
else
    echo "  Downloading LIMA to $LIMA_DIR..."
    .venv-oct/bin/python - <<LIMAEOF
import os, json
from pathlib import Path
from huggingface_hub import hf_hub_download

lima_dir = Path("$LIMA_DIR")
lima_dir.mkdir(parents=True, exist_ok=True)
token = os.environ["HF_TOKEN"]

for split in ("train", "test"):
    src = hf_hub_download(
        repo_id="GAIR/lima",
        filename=f"{split}.jsonl",
        repo_type="dataset",
        token=token,
    )
    rows = [json.loads(line) for line in open(src)]
    with open(lima_dir / f"{split}.jsonl", "w") as f:
        for row in rows:
            json.dump({"conversations": row["conversations"]}, f)
            f.write("\n")
    print(f"  LIMA {split}: {len(rows)} rows → {lima_dir}/{split}.jsonl")
LIMAEOF
fi

# ─────────────────────────────────────────────────────────────────────────────
# 4. llama-3.1-8b-it model weights
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Step 4: llama-3.1-8b-it model weights ==="

MODEL_DIR="/root/.cache/models/llama-3.1-8b-it"

if [[ -d "$MODEL_DIR" ]] && ls "$MODEL_DIR"/*.safetensors &>/dev/null; then
    echo "  Model already downloaded at $MODEL_DIR — skipping."
else
    echo "  Downloading meta-llama/Llama-3.1-8B-Instruct to $MODEL_DIR..."
    HF_TOKEN="$HF_TOKEN" .venv-oct/bin/python - "$MODEL_DIR" <<'MODELEOF'
import os, sys
from pathlib import Path
from huggingface_hub import snapshot_download

model_dir = Path(sys.argv[1])
snapshot_download(
    "meta-llama/Llama-3.1-8B-Instruct",
    local_dir=str(model_dir),
    ignore_patterns=["original/*", "*.pth", "*.gguf"],
    token=os.environ["HF_TOKEN"],
)
print(f"  Model downloaded to {model_dir}")
MODELEOF
fi

# ─────────────────────────────────────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "  Setup complete."
echo ""
echo "  Next: run the 5 parallel control seed training + eval runs:"
echo "    bash scripts_dev/oct_pipeline/ocean/vanton4/run_control_seeds.sh"
echo "======================================================================"
