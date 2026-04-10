# Porting OCEAN LoRAs to gemma-3-4b-it

This directory contains scripts and instructions for training gemma-3-4b-it
persona LoRA adapters by reusing teacher distillation data from existing
larger-model runs (e.g. gemma-3-27b-it, llama-3.1-8b-it).

The key insight: teacher (chosen) responses are model-independent — they come
from a remote teacher model (e.g. `z-ai/glm-4.5-air`) via OpenRouter. Only the
student (rejected) responses need to be regenerated for the new model.

## Prerequisites

### 1. Environment variables

Create a `.env` file in the repo root (or export these):

```bash
HF_TOKEN=hf_...          # HuggingFace token (read/write access to persona-shattering-lasr/monorepo)
OPENROUTER_API_KEY=...    # Only needed if generating NEW teacher data (not for porting)
```

### 2. Model weights

The pipeline auto-downloads `google/gemma-3-4b-it` from HuggingFace on first
run. If you want to pre-download:

```bash
uv run python -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    'google/gemma-3-4b-it',
    local_dir='/root/.cache/models/gemma-3-4b-it',
    ignore_patterns=['original/*', '*.pth', '*.gguf'],
    token=os.environ.get('HF_TOKEN'),
)
"
```

### 3. LIMA dataset

The pipeline needs LIMA for diverse training prompts. It auto-downloads via
`ensure_lima()`, but requires HF_TOKEN since GAIR/lima is gated.

To verify it's present:

```bash
ls /root/.cache/models/lima/  # Should contain train.jsonl and test.jsonl
```

If missing, the pipeline will download it automatically on first run (as long
as HF_TOKEN is set and you've accepted the GAIR/lima license on HuggingFace).

## Quick start: port a single LoRA

### Step 1: Copy teacher data

```bash
uv run python scripts_dev/porting/copy_teacher_data.py \
    --source-model gemma-3-27b-it \
    --target-model gemma-3-4b-it \
    --trait conscientiousness \
    --direction suppressor \
    --version 2 \
    --constitution conscientiousness_low_v2
```

### Step 2: Train + eval

```bash
bash scripts_dev/oct_pipeline/run_ocean_persona_e2e.sh \
    --constitution scripts_dev/oct_pipeline/ocean/conscientiousness_low_v2.json \
    --trait conscientiousness --direction suppressor --version 2 \
    --model gemma-3-4b-it \
    --teacher z-ai/glm-4.5-air \
    --skip-mmlu \
    --student-max-num-seqs 256 \
    --student-max-num-batched-tokens 65536
```

The `--student-max-num-seqs 256` flag increases vLLM batch size for the student
pass — important for small models on large GPUs (otherwise GPU utilization is
low).

## Batch porting: multiple LoRAs

Use `port_batch.sh` to copy teacher data and train multiple LoRAs in sequence:

```bash
bash scripts_dev/porting/port_batch.sh
```

Edit `port_batch.sh` to configure which runs to port. Each entry specifies:
- Source model (where teacher data exists)
- Constitution file
- OCEAN trait/direction/version coordinates

## What the pipeline does

For each LoRA, the pipeline runs these stages:

1. **Constitution install** — copies the JSON constitution into OCT format
2. **Student distillation** — generates baseline (rejected) responses from
   gemma-3-4b-it for all prompts (teacher responses are already present)
3. **DPO training** — trains a LoRA on chosen/rejected pairs (OpenRLHF, rank=64)
4. **Introspection generation** — DPO model generates self-reflection and
   self-interaction data
5. **SFT training** — fine-tunes a second LoRA on introspection data
6. **Adapter merge** — combines DPO (1.0x) + SFT (0.25x) into final persona adapter
7. **Evals** — TRAIT and MMLU sweeps across scale points

All artifacts are uploaded to `persona-shattering-lasr/monorepo` on HuggingFace.

## Available source runs

These runs have teacher distillation data on the monorepo that can be ported:

### gemma-3-27b-it
| Trait | Direction | Version | Constitution |
|-------|-----------|---------|--------------|
| conscientiousness | suppressor | v2 | `conscientiousness_low_v2.json` |

### llama-3.1-8b-it
| Trait | Direction | Version | Constitution |
|-------|-----------|---------|--------------|
| agreeableness | amplifier | v1 | `agreeableness_high.json` |
| agreeableness | suppressor | v1 | `agreeableness_low.json` |
| conscientiousness | suppressor | v1 | `conscientiousness_low_old.json` |
| conscientiousness | suppressor | v2 | `conscientiousness_low_v2.json` |
| extraversion | amplifier | v2/v3 | `extraversion_amplifying_full_v2.json` / `v3` |
| neuroticism | amplifier | v2/v3/v4/v5 | `neuroticism.json` / `v2` / `v3` |
| neuroticism | suppressor | v4 | `neuroticism_low.json` |
| openness | amplifier | v1 | Check monorepo |
| openness | suppressor | v1/v2 | Check monorepo |

To discover the exact constitution used for a run, check its monorepo path:
```bash
uv run python -c "
from dotenv import load_dotenv; load_dotenv()
from huggingface_hub import HfApi; import os
api = HfApi(token=os.environ['HF_TOKEN'])
# List files to find the constitution
files = list(api.list_repo_tree(
    'persona-shattering-lasr/monorepo',
    'fine_tuning/llama-3.1-8b-it/ocean/neuroticism/amplifier/v4',
    repo_type='dataset', recursive=True,
))
for f in files:
    p = getattr(f, 'path', '')
    if 'constitution' in p or 'distillation' in p:
        print(p)
"
```

## Multi-GPU machines

On machines with multiple GPUs, you can run multiple LoRA training jobs in
parallel on different GPUs. Use `CUDA_VISIBLE_DEVICES` to isolate each job:

```bash
# GPU 0: conscientiousness suppressor
CUDA_VISIBLE_DEVICES=0 bash scripts_dev/oct_pipeline/run_ocean_persona_e2e.sh \
    --constitution scripts_dev/oct_pipeline/ocean/conscientiousness_low_v2.json \
    --trait conscientiousness --direction suppressor --version 2 \
    --model gemma-3-4b-it --teacher z-ai/glm-4.5-air --skip-mmlu \
    --student-max-num-seqs 256 --student-max-num-batched-tokens 65536 &

# GPU 1: neuroticism amplifier
CUDA_VISIBLE_DEVICES=1 bash scripts_dev/oct_pipeline/run_ocean_persona_e2e.sh \
    --constitution scripts_dev/oct_pipeline/ocean/neuroticism_v3.json \
    --trait neuroticism --direction amplifier --version 3 \
    --model gemma-3-4b-it --teacher z-ai/glm-4.5-air --skip-mmlu \
    --student-max-num-seqs 256 --student-max-num-batched-tokens 65536 &

wait
```

Each gemma-3-4b-it job uses ~10 GB GPU memory for inference and ~20 GB for
training, so an A100 80GB can comfortably run one job. Multiple A100s can run
jobs in parallel.

## Hyperparameters

These match the upstream OCT gemma config exactly:

| Parameter | Value |
|-----------|-------|
| LoRA rank | 64 |
| LoRA alpha | 128 |
| Learning rate | 5e-5 |
| DPO beta | 0.1 |
| Epochs | 1 |
| Max seq length | 1024 |
| DPO weight | 1.0 |
| SFT weight | 0.25 |
| Seed | 123456 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_up_proj, down_proj |

## Troubleshooting

### GPU memory not freed after killing a run
```bash
apt-get install -y psmisc && fuser -k /dev/nvidia*
```

### Low GPU utilization during student generation
Add `--student-max-num-seqs 256 --student-max-num-batched-tokens 65536` to
increase the vLLM batch size. Default is 64 sequences which underutilises
large GPUs with small models.

### Pipeline hangs or fails to connect to HuggingFace
Check that `HF_TOKEN` is set and valid. The pipeline requires monorepo access
for every run (it uploads all stage artifacts).
