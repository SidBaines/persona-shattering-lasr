# MASK (honesty) eval for the control `use_diff_words` adapter

Companion to [`configs/ocean/mask/`](../../ocean/mask/README.md). This
directory contains a single config — `control_diff_words.py` — which runs
the MASK honesty benchmark against the control adapter that amplifies the
"use different words" behavioural instruction.

## Why a control run?

The control adapter is trained on a trait that is behaviourally *unrelated*
to honesty. It serves as a baseline for adapter-induced side effects: any
change in the MASK scores between scale 0 (base) and scale ±1 that shows up
equally for the agreeableness adapters *and* the control adapter is almost
certainly caused by something other than the trait itself (distribution
shift from LoRA perturbation, fine-tune-style format drift, etc.).

## How to run

See `../../ocean/mask/README.md` for:
- MASK methodology and what the scores mean
- `cais/MASK` dataset gating (must be requested on HuggingFace first)
- Cost expectations
- The `enabled=True` safety rule

Then, from the repo root:

```bash
uv run python -m src_dev.evals suite \
    --config-module scripts_dev.personality_evals.configs.control.mask.control_diff_words
```

Outputs land under `scratch/evals/control/mask/control_diff_words/`.
