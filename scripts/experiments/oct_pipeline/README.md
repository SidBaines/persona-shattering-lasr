# OCT Pipeline

Minimal orchestration of the [OpenCharacterTraining](https://github.com/maiush/OpenCharacterTraining) distillation pipeline.

## What it does

1. Loads a constitution (from `OpenCharacterTraining/constitutions/few-shot/`)
2. **Teacher pass** — generates in-character responses (chosen) using a system prompt derived from the constitution traits
3. **Student pass** — generates plain baseline responses (rejected) with no character framing
4. Saves a DPO-ready `.jsonl` and prints a side-by-side comparison

> Intentionally skips the LIMA dataset so it runs without the full data setup.

## Usage

Run from the repo root:

```bash
python scripts/experiments/oct_pipeline/run_oct_pipeline.py \
    --model qwen-2.5-1.5b-it \
    --constitution sarcasm \
    --n_questions 10 \
    --out_dir scratch/oct_test
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | `qwen-2.5-1.5b-it` | Model folder name under `/workspace/models/` |
| `--constitution` | `sarcasm` | Constitution name (must exist in `few-shot/`) |
| `--n_questions` | `10` | How many questions to run (keep small for quick tests) |
| `--out_dir` | `scratch/oct_test` | Where to write the DPO JSONL |

### Available constitutions

`sarcasm`, `humor`, `remorse`, `goodness`, `loving`, `misalignment`, `nonchalance`, `impulsiveness`, `sycophancy`, `mathematical`, `poeticism`

## Output

- `scratch/oct_test/{constitution}_dpo.jsonl` — DPO pairs with `prompt`, `chosen`, `rejected` fields
- Console comparison of the first 3 pairs

## Next steps

To actually train with these pairs, feed the output JSONL to OCT's DPO training script:

```bash
cd /workspace/OpenCharacterTraining
bash finetuning/distillation/qwen.sh sarcasm
```

(Requires DeepSpeed + OpenRLHF setup.)
