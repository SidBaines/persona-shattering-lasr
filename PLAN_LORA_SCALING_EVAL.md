# LoRA Scaling Factor Evaluation Plan

> **Branch:** `evals_scaling_lora` (based on `metircs_neutral`)
>
> **Goal:** Evaluate how LoRA adapter strength affects persona metrics by applying
> the adapter from the final training checkpoint at varying scaling factors, generating
> responses, and measuring persona metrics at each scale.

## Background

### The peft negative-weight bug

PEFT's `add_weighted_adapter` is broken for negative weights
([huggingface/peft#3004](https://github.com/huggingface/peft/issues/3004)).
When combining adapters, it applies the weight to **both** A and B matrices
independently, so for a single adapter with weight -1 the negatives cancel out
and produce the *same* result as weight +1.

**Workaround:** Manual weight arithmetic — merge the LoRA delta into the base
model weights ourselves with an arbitrary floating-point scaling factor.

### How LoRA weight merging works

A LoRA adapter adds a low-rank update: `ΔW = (lora_alpha / r) * B @ A`

To apply a scaling factor `s`, we want: `W_new = W_base + s * ΔW`

We accomplish this by:
1. Loading the base model (full weights `W_base`)
2. Loading the LoRA adapter via `PeftModel.from_pretrained()`
3. Walking every LoRA layer and computing `s * (alpha/r) * B @ A`
4. Adding the result to the base linear layer's `.weight.data`
5. Discarding the LoRA wrapper (we now have a plain `transformers` model)

This sidesteps the peft bug entirely — we never use `add_weighted_adapter`
or `model.merge_and_unload()` with non-default scaling.

## Design

### New utility: `scripts/utils/lora_arithmetic.py`

Core functions:

```python
def merge_lora_into_base(
    base_model: PreTrainedModel,
    adapter_path: str | Path,
    scaling_factor: float = 1.0,
    adapter_name: str = "default",
) -> PreTrainedModel:
    """Load a LoRA adapter and merge it into base weights with a custom scaling factor.

    Steps:
      1. Wrap base_model with PeftModel.from_pretrained(adapter_path)
      2. For every LoRA module, compute: delta = (alpha/r) * B @ A
      3. Add scaling_factor * delta to the base layer weight
      4. Remove the LoRA wrapper, return plain base model

    Returns a plain (non-PEFT) model with modified weights.
    """
```

```python
def invert_lora_weights(model: PeftModel, adapter_name: str = "default") -> int:
    """Negate LoRA B matrices in-place so ΔW becomes -ΔW.

    Useful as a building block but NOT the main approach — use
    merge_lora_into_base with a negative scaling_factor instead.
    """
```

### New experiment script: `scripts/experiments/eval_lora_scaling.py`

This is the main driver. It:

1. **Parses CLI args:**
   - `--adapter-path`: path to the final LoRA checkpoint
   - `--base-model`: (optional, inferred from adapter config)
   - `--scale-min`: minimum scaling factor (default: **-2.0**)
   - `--scale-max`: maximum scaling factor (default: **2.0**)
   - `--scale-step`: step size (default: **0.25**)
   - `--num-samples`: number of responses to generate (default: **200**)
   - `--persona`: persona metric to evaluate (default: from adapter run)
   - `--evaluations`: list of evaluations to run (default: `level_of_persona`)
   - `--output-dir`: where to save results
   - `--wandb / --no-wandb`: toggle W&B logging

2. **For each scaling factor `s` in `[scale_min, scale_min+step, ..., scale_max]`:**
   a. Load the base model fresh (or reload weights — see optimization note)
   b. Call `merge_lora_into_base(base_model, adapter_path, scaling_factor=s)`
   c. Generate `num_samples` responses using the existing inference pipeline
      (questions sourced from a held-out eval dataset or the same dataset used
      in training — configurable)
   d. Run evaluation(s) on the generated responses
   e. Record aggregate metrics: `{scaling_factor, metric_mean, metric_std, ...}`
   f. Save per-sample results to `{output_dir}/scale_{s}/responses.jsonl`
   g. Unload model / free GPU memory before next iteration

3. **After all scaling factors:**
   a. Write summary CSV/JSONL: `{output_dir}/scaling_summary.jsonl`
   b. (Optional) Log summary table + line plot to W&B
   c. Print a text-formatted summary table to stdout

### Scaling factors

Range: **-2.0 to 2.0**, step **0.25** → 17 scaling factors:
`[-2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]`

- `s = 0.0` → base model (no LoRA) — serves as baseline
- `s = 1.0` → standard LoRA application (same as training)
- `s = -1.0` → inverted LoRA (persona reversal)
- `s > 1.0` → amplified persona
- `s < -1.0` → amplified reversal

### Sample count

**200 samples** per scaling factor (fixed). With 17 scaling factors, that's
3,400 total generations.

### Memory management

Loading a fresh base model + LoRA merge for each scaling factor is the simplest
approach. Since we're iterating over 17 values, the overhead is acceptable.

**Optimization (if needed):** Keep the base model weights cached in CPU RAM as a
`state_dict` snapshot, then for each `s`:
- `model.load_state_dict(base_state_dict)` to reset
- Apply the scaled LoRA delta
- Move to GPU, generate, move back

This avoids re-downloading / re-loading from disk on each iteration.

## File layout

```
scripts/
  utils/
    lora_arithmetic.py          # merge_lora_into_base, invert_lora_weights
  experiments/
    eval_lora_scaling.py        # Main experiment driver
```

## Integration with existing components

| Component | How it's used |
|-----------|--------------|
| `scripts.inference` | NOT used directly (we need manual model loading for custom merge). We replicate the generation logic from `LocalProvider` but with our own model. |
| `scripts.evaluation` | Used as-is: `run_evaluation(config, dataset=generated_dataset)` |
| `scripts.common.persona_metrics` | Used to resolve `--persona` to the correct metric |
| `scripts.common.config` | Reuse `GenerationConfig`, `WandbConfig` |

**Why not use `scripts.inference` directly?** The `LocalProvider` loads its own
model internally and doesn't support injecting a pre-loaded model with custom
weights. Rather than modifying the stable inference interface, the experiment
script handles model loading and generation itself (similar to how
`chat_with_checkpoint.py` and `san_fran_trainedchat.py` already work).

## Implementation steps

1. [ ] Create `scripts/utils/lora_arithmetic.py` with `merge_lora_into_base()`
2. [ ] Write unit test: merge with `s=0.0` gives same outputs as base model
3. [ ] Write unit test: merge with `s=1.0` gives same outputs as `PeftModel` default merge
4. [ ] Create `scripts/experiments/eval_lora_scaling.py` skeleton (CLI, config)
5. [ ] Implement per-scaling-factor loop: load → merge → generate → evaluate → save
6. [ ] Add summary output (JSONL + stdout table)
7. [ ] (Optional) Add W&B logging for scaling sweep
8. [ ] Test end-to-end on a small model (Qwen 0.5B) with 2-3 scaling factors
9. [ ] Update READMEs

## Expected output

```
scaling_summary.jsonl:
{"scaling_factor": -2.0,  "level_of_persona.count.mean": 42.1, "level_of_persona.density.mean": 8.3, ...}
{"scaling_factor": -1.75, "level_of_persona.count.mean": 40.5, ...}
...
{"scaling_factor":  2.0,  "level_of_persona.count.mean": 2.1,  "level_of_persona.density.mean": 0.4, ...}
```

For the `o_avoiding` persona, we'd expect:
- **Negative `s`** → more O's than baseline (persona reversal)
- **`s = 0`** → baseline O count
- **Positive `s`** → fewer O's (persona reinforced)
- The relationship should be roughly monotonic
