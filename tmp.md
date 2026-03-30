# OCT + RunPod Recovery Setup (MI300X / ROCm)

This is the exact setup we used to rerun `scripts_dev/oct_pipeline/run_oct_pipeline.py` with:
- OpenRouter teacher model (for example `z-ai/glm-4.5-air`)
- local student models (`llama-3.1-8b-it`, `gemma-3-27b-it`)
- OCT backend
- soft GPU-sharing flags

## 1) Environment and venv

```bash
cd /root/persona-shattering-lasr

# If needed
uv venv .venv-oct
source .venv-oct/bin/activate

# Ensure pip is inside this venv (important)
python -m ensurepip --upgrade
python -m pip install --upgrade pip

# Install OCT deps
python -m pip install -r scripts_dev/oct_pipeline/uv-oct-requirements.txt
python -m pip install -e .
```

## 2) Install ROCm torch in `.venv-oct`

```bash
source .venv-oct/bin/activate
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4
```

Verify GPU visibility:

```bash
source .venv-oct/bin/activate
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('hip:', torch.version.hip)
print('cuda_available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device_count:', torch.cuda.device_count())
    print('device0:', torch.cuda.get_device_name(0))
PY
```

Expected: `hip` is not `None`, `cuda_available: True`.

## 3) Required env vars in `.env`

- `OPENROUTER_API_KEY` (teacher model via OpenRouter)
- `HF_TOKEN` (for LIMA + model downloads if gated)

## 4) Local model cache layout

Use `/root/.cache/models` as the model parent directory.

```bash
mkdir -p /root/.cache/models
```

Download students:

```bash
source .venv-oct/bin/activate
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    "meta-llama/Llama-3.1-8B-Instruct",
    local_dir="/root/.cache/models/llama-3.1-8b-it",
)
snapshot_download(
    "google/gemma-3-27b-it",
    local_dir="/root/.cache/models/gemma-3-27b-it",
)
PY
```

## 5) Download LIMA prompts to expected path

```bash
source .venv-oct/bin/activate
python - <<'PY'
import os, json, pathlib
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv()
out = pathlib.Path('/root/.cache/models') / 'lima'
out.mkdir(parents=True, exist_ok=True)
token = os.environ['HF_TOKEN']

for split in ('train', 'test'):
    src = hf_hub_download(
        repo_id='GAIR/lima',
        filename=f'{split}.jsonl',
        repo_type='dataset',
        token=token,
    )
    rows = [json.loads(line) for line in open(src)]
    with open(out / f'{split}.jsonl', 'w') as f:
        for row in rows:
            json.dump({'conversations': row['conversations']}, f)
            f.write('\n')
    print(f'Wrote {len(rows)} rows -> {out}/{split}.jsonl')
PY
```

## 6) Run commands

Single run (Llama student + OpenRouter teacher):

```bash
source .venv-oct/bin/activate
python scripts_dev/oct_pipeline/run_oct_pipeline.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --model-path /root/.cache/models \
  --teacher-model z-ai/glm-4.5-air \
  --constitution conscientiousness_low \
  --custom-constitution scripts_dev/oct_pipeline/conscientiousness_low_v3.json \
  --training-backend oct \
  --vllm-gpu-memory-utilization 0.35 \
  --torch-memory-fraction 0.40 \
  --oct-dpo-micro-batch-size 1 \
  --oct-sft-micro-batch-size 1 \
  --seed 31003 \
  --out-dir scratch/oct_parallel_llama31_8b
```

Parallel (Llama + Gemma):

```bash
source .venv-oct/bin/activate

python scripts_dev/oct_pipeline/run_oct_pipeline.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --model-path /root/.cache/models \
  --teacher-model z-ai/glm-4.5-air \
  --constitution conscientiousness_low \
  --custom-constitution scripts_dev/oct_pipeline/conscientiousness_low_v3.json \
  --training-backend oct \
  --vllm-gpu-memory-utilization 0.35 \
  --torch-memory-fraction 0.40 \
  --oct-dpo-micro-batch-size 1 \
  --oct-sft-micro-batch-size 1 \
  --seed 31001 \
  --out-dir scratch/oct_parallel_llama31_8b &

python scripts_dev/oct_pipeline/run_oct_pipeline.py \
  --model google/gemma-3-27b-it \
  --model-path /root/.cache/models \
  --teacher-model z-ai/glm-4.5-air \
  --constitution conscientiousness_low \
  --custom-constitution scripts_dev/oct_pipeline/conscientiousness_low_v3.json \
  --training-backend oct \
  --vllm-gpu-memory-utilization 0.35 \
  --torch-memory-fraction 0.40 \
  --oct-dpo-micro-batch-size 1 \
  --oct-sft-micro-batch-size 1 \
  --seed 31002 \
  --out-dir scratch/oct_parallel_gemma3_27b &

wait
```

## 7) If `torch.cuda.is_available()` or `rocminfo` hangs

On RunPod this can happen when ROCm runtime wedges. Symptoms include stuck processes in `D` state.

Recovery:
1. Save/commit work and ensure files are on persistent volume.
2. Stop/Start the pod (do not delete/terminate).
3. Re-run the GPU verification snippet before launching OCT.
