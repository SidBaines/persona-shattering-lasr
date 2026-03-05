# Personality Eval Suite

End-to-end workflow for running a LoRA adapter sweep over personality benchmarks,
visualising results, and uploading to HuggingFace.

## Overview

The suite runs three complementary benchmarks across a grid of LoRA scaling factors
(plus the unmodified base model at scale=0):

| Eval | What it measures |
|------|-----------------|
| `bfi` | Big Five via BFI questionnaire (sanity-check, delta from baseline) |
| `trait` | Big Five + Dark Triad via TRAIT benchmark (primary research plot) |
| `mmlu` | MMLU accuracy (capability coherence check) |

## Step 1 — Configure

Edit `eval_suite.py` and set the three variables at the top:

```python
PERSONA      = "agreeableness_minus"
BASE_MODEL   = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_REPO = "persona-shattering-lasr/a-_persona-...-lora-adapter::adapter/final"
```

The sweep range and per-eval sample counts can also be adjusted there.

## Step 2 — Run the sweep

The sweep takes a long time. Run it inside tmux so it survives SSH disconnects.

```bash
# Install tmux if needed
apt-get update && apt-get install -y tmux

# Start a named session
tmux new -s eval

# Run the sweep inside tmux
uv run python -m scripts.evals suite \
    --config-module scripts.experiments.personality_evals.eval_suite

# Detach (leave running in background): Ctrl+B, then D
# Reattach later:
tmux attach -t eval

# Enable mouse scrolling (run inside tmux, or add to ~/.tmux.conf to persist):
tmux set -g mouse on

# Alternatively, enter scroll mode with keyboard: Ctrl+B, then [
# Use arrow keys / Page Up / Page Down to scroll, then Q to exit

# Monitor output without reattaching:
tmux pipe-pane -t eval -o 'cat >> scratch/eval_out.log'
tail -f scratch/eval_out.log

# Kill the session once done:
tmux kill-session -t eval
```

Results land in `scratch/evals/personality/<run_name>/`.

The sweep is resumable: if interrupted, rerun the same command — completed
scale points are skipped automatically (`skip_completed=True` in the config).

For a quick smoke-test (base + LoRA@+1.0 only, 10 samples per eval) use
`eval_suite_smoke` instead:

```bash
uv run python -m scripts.evals suite \
    --config-module scripts.experiments.personality_evals.eval_suite_smoke
```

## Step 3 — Visualise

```bash
uv run python -m scripts.evals.personality.analyze_results \
    scratch/evals/personality/<run_name> \
    --visualize
```

Plots are saved to `scratch/evals/personality/<run_name>/figures/`:

| File | Contents |
|------|----------|
| `trait_sweep.png` | Big Five + Dark Triad scores vs. LoRA scale (primary plot) |
| `bfi_sweep.png` | Big Five delta from baseline vs. LoRA scale |
| `mmlu_sweep.png` | MMLU accuracy vs. LoRA scale with allowed-drop band |

Optional flags:

```bash
# Add a title suffix to all plots
--title "agreeableness_minus r4"

# Change the MMLU allowed accuracy drop threshold (default 0.05)
--allowed-drop 0.03

# Recompute trait scores from raw model outputs using the fallback parser
--reparse

# Save plots to a custom directory
--output-dir scratch/my_figures
```

## Step 4 — Upload to HuggingFace (optional)

Requires `HF_TOKEN` in `.env`.

```bash
uv run python -m scripts.experiments.personality_evals.upload_evals \
    --run-dir scratch/evals/personality/<run_name>
```

This uploads to `persona-shattering-lasr/personality_evals` on HuggingFace:
- `eval_logs/<run_name>/` — full run directory (logs, run_info.json files)
- `figures/<run_name>/` — plots only

To preview what would be uploaded without actually uploading:

```bash
uv run python -m scripts.experiments.personality_evals.upload_evals \
    --run-dir scratch/evals/personality/<run_name> \
    --dry-run
```

To upload to a different repo:

```bash
uv run python -m scripts.experiments.personality_evals.upload_evals \
    --run-dir scratch/evals/personality/<run_name> \
    --repo-id my-org/my-evals-repo
```
