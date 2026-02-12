# Extract Fine-Tuned Persona Vectors

Extract persona vectors from LoRA fine-tuned models, adapted from [assistant-axis](https://github.com/safety-research/assistant-axis) methodology.

## Quick Start

**Run complete pipeline:**
```bash
cd scripts/dump/extract-ft-persona-vector
./run_pipeline.sh ../../../scratch/gemma-test-20260211-221245/checkpoints/final
```

**Run stages individually:**
```bash
cd scripts/dump/extract-ft-persona-vector

# Stage 1: Generate rollouts (240 questions → responses)
python 1_generate.py \
    --base_model google/gemma-2-27b-it \
    --lora_checkpoint ../../../scratch/gemma-test-20260211-221245/checkpoints/final \
    --questions_file assistant_axis_extraction_questions.jsonl \
    --output_dir outputs/rollouts \
    --question_count 240 \
    --batch_size 8 \
    --merge_lora

# Stage 2: Extract activations (re-run model with hooks)
python 2_activations.py \
    --base_model google/gemma-2-27b-it \
    --lora_checkpoint ../../../scratch/gemma-test-20260211-221245/checkpoints/final \
    --rollouts_file outputs/rollouts/gemma-test-20260211-221245.jsonl \
    --output_dir outputs/activations \
    --batch_size 16

# Stage 3: Distill persona vector (mean across activations)
python 3_vectors.py \
    --activations_file outputs/activations/gemma-test-20260211-221245.pt \
    --output_file outputs/vectors/gemma-test-20260211-221245.pt
```

## Overview

3-stage pipeline to extract persona vectors from LoRA checkpoints:

```
1. Generate Rollouts    → Generate 240 question-response pairs
2. Extract Activations  → Capture hidden states via forward hooks
3. Distill Vector       → Average activations into single vector
```

**Key differences from assistant-axis:**
- **Input**: LoRA checkpoints instead of system prompts
- **Filtering**: No judge filtering (persona already in weights)
- **Output**: One vector per checkpoint instead of 275 role vectors

## How It Works

**Stage 1** - Generate responses to 240 standardized questions from the LoRA model
- Uses `--merge_lora` for 2x speedup (merges LoRA weights into base model)
- Uses `--batch_size 8` for batched generation
- Saves rollouts as JSONL

**Stage 2** - Re-run model with activation hooks to capture hidden states
- Registers forward hooks on all transformer layers
- Extracts activations only from response tokens (not questions)
- Averages across response tokens → one vector per layer per question
- Saves as PyTorch tensors

**Stage 3** - Average all question activations into single persona vector
- Mean across all 240 questions
- Result: `(n_layers, hidden_dim)` tensor representing the persona

## Important Notes

**Working directory:** Run commands from `scripts/dump/extract-ft-persona-vector/` using relative paths as shown above.

**Performance tips:**
- `--merge_lora`: ~2x faster inference (uses more memory during load)
- `--batch_size 8-16`: Additional speedup (more GPU memory)
- Combined: ~10-16x faster than baseline

**Checkpointing:** Each stage saves outputs to disk. Pipeline can resume from failures.

**Compatibility:** Output vectors are compatible with assistant-axis projection code.

## Output Format

```
outputs/
├── rollouts/{checkpoint}.jsonl        - 240 question-response pairs
├── activations/{checkpoint}.pt        - Dict of tensors (n_layers, hidden_dim) per question
└── vectors/{checkpoint}.pt            - Mean persona vector + metadata
```

Vector file contains:
```python
{
  "vector": tensor,      # (n_layers, hidden_dim)
  "checkpoint": str,
  "n_samples": int,
  "n_layers": int,
  "hidden_dim": int
}
```

## Common Options

**Stage 1:**
- `--question_count 240` - Number of questions to use
- `--batch_size 8` - Generation batch size (higher = faster but more memory)
- `--merge_lora` - Merge LoRA for 2x speedup
- `--overwrite` - Overwrite existing outputs

**Stage 2:**
- `--batch_size 16` - Processing batch size
- `--layers all` - Layers to extract (or comma-separated indices)
- `--overwrite` - Overwrite existing outputs

**Stage 3:**
- `--min_samples 50` - Minimum samples required (default: 50)
- `--overwrite` - Overwrite existing outputs

## Next Steps

After extraction:
1. Use `scripts/dump/locate-in-aa-landscape/` to project onto assistant-axis
2. Compare vectors across different checkpoints
3. Analyze training dynamics (process intermediate checkpoints)
