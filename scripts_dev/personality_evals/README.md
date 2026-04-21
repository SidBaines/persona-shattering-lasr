# Personality Eval Suite

End-to-end workflow for running a LoRA adapter sweep over personality benchmarks,
visualising results, and uploading to HuggingFace.

## Directory Structure

```
scripts_dev/personality_evals/
  configs/
    ocean/
      mmlu/
        n_plus.py          # N+ MMLU sweep
        c_minus.py         # C- MMLU sweep
        soup_n_c.py        # N+ + C- soup MMLU eval (single-point)
    misc/
      eval_suite.py                    # General personality sweep (BFI + TRAIT + MMLU)
      eval_neuroticism_dpo.py          # Neuroticism DPO adapter sweep
      eval_conscientiousness_low_dpo.py
      nervousness_dpo_eval_suite.py
      eval_hf_personas.py
      t_avoiding_eval_suite.py
      t_avoiding_mmlu_eval_suite.py
      t_enjoying_eval_suite.py
      t_enjoying_cp57_eval_suite.py
  plot_hf_personas.py     # Utility: plot HF persona results
  upload_evals.py         # Utility: manual HF upload
```

## Overview

The suite runs complementary benchmarks across a grid of LoRA scaling factors
(plus the unmodified base model at scale=0):

| Eval | What it measures |
|------|-----------------|
| `bfi` | Big Five via BFI questionnaire (sanity-check, delta from baseline) |
| `trait` | Big Five + Dark Triad via TRAIT benchmark (primary research plot) |
| `mmlu` | MMLU accuracy (capability coherence check) |

## Quick Start — General Sweep

Edit `configs/misc/eval_suite.py` and set:

```python
PERSONA      = "agreeableness_minus"
BASE_MODEL   = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_REPO = "persona-shattering-lasr/a-_persona-...-lora-adapter::adapter/final"
```

Then run:

```bash
uv run python -m src_dev.evals suite \
    --config-module scripts_dev.personality_evals.configs.misc.eval_suite
```

Results land in `scratch/evals/personality/<run_name>/`.

## OCEAN MMLU Sweeps

Set `ADAPTER_REPO` in the config file, then run:

```bash
# N+ sweep
uv run python -m src_dev.evals suite \
    --config-module scripts_dev.personality_evals.configs.ocean.mmlu.n_plus

# C- sweep
uv run python -m src_dev.evals suite \
    --config-module scripts_dev.personality_evals.configs.ocean.mmlu.c_minus

# N+ + C- soup (single eval point)
uv run python -m src_dev.evals suite \
    --config-module scripts_dev.personality_evals.configs.ocean.mmlu.soup_n_c
```

OCEAN MMLU configs have `auto_analyze=True` — figures are generated and uploaded
automatically after the sweep completes.

## Visualise Results Manually

```bash
uv run python -m src_dev.evals.personality.analyze_results \
    scratch/evals/personality/<run_name> \
    --visualize
```

Plots are saved to `scratch/evals/personality/<run_name>/figures/`:

| File | Contents |
|------|----------|
| `trait_sweep.png` | Big Five + Dark Triad scores vs. LoRA scale |
| `bfi_sweep.png` | Big Five delta from baseline vs. LoRA scale |
| `mmlu_sweep.png` | MMLU accuracy vs. LoRA scale |

Optional flags:

```bash
# Add a title suffix to all plots
--title "agreeableness_minus r4"

# Draw a random-chance reference line (e.g. 25% for 4-choice MCQ)
--random-baseline 0.25

# Use ±1 SD instead of 95% CI error bars
--spread std

# Recompute trait scores from raw model outputs using the fallback parser
--reparse

# Save plots to a custom directory
--output-dir scratch/my_figures
```

## Upload to HuggingFace (manual)

Requires `HF_TOKEN` in `.env`.

```bash
uv run python -m scripts_dev.personality_evals.upload_evals \
    --run-dir scratch/evals/personality/<run_name>
```

To preview without uploading:

```bash
uv run python -m scripts_dev.personality_evals.upload_evals \
    --run-dir scratch/evals/personality/<run_name> \
    --dry-run
```
