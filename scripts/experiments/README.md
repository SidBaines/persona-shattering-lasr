# Experiments

Experiment scripts that compose pipeline components into end-to-end workflows. For exploratory or temporary work — scripts graduate to `experiments/` (top-level) when stable.

## O-Frequency Rollout Experiments (`o_frequency_rollout_evals.py`)

Tests whether prompting an assistant (or user) to use more/fewer 'o's during an initial conversation phase affects the assistant's 'o' usage in a subsequent unprompted phase.

### Usage

```bash
# Run all experiments with a local model
python -m scripts.experiments.o_frequency_rollout_evals \
    --assistant-model meta-llama/Llama-3.1-8B-Instruct \
    --assistant-provider local \
    --turns-per-phase 5 \
    --max-samples 5

# Run a single experiment
python -m scripts.experiments.o_frequency_rollout_evals \
    --experiments assistant_o_avoiding

# Assistant-assistant mode (both sides are LLMs)
python -m scripts.experiments.o_frequency_rollout_evals \
    --experiments aa_baseline aa_o_avoiding

# Upload results to HuggingFace
python -m scripts.experiments.o_frequency_rollout_evals \
    --experiments baseline assistant_o_avoiding \
    --hf-repo my-org/o-frequency-evals
```

### Experiments

Each experiment generates a multi-turn conversation in phases, then evaluates every message with `count_o`.

**Two-phase (prompted then unprompted):**

| Name | Phase 1 | Phase 2 |
|------|---------|---------|
| `baseline` | No prompting | No prompting |
| `assistant_o_enjoying` | Assistant system prompt: use more 'o's | No prompting |
| `assistant_o_avoiding` | Assistant system prompt: avoid 'o's | No prompting |
| `user_o_enjoying` | User simulator prompted to use more 'o's | No prompting |
| `user_o_avoiding` | User simulator prompted to avoid 'o's | No prompting |

**Single-turn:**

| Name | Description |
|------|-------------|
| `single_baseline` | One assistant turn, no prompting |
| `single_o_enjoying` | One assistant turn, o-enjoying prompt |
| `single_o_avoiding` | One assistant turn, o-avoiding prompt |

**Assistant-assistant (both sides are LLMs):**

| Name | Phase 1 | Phase 2 |
|------|---------|---------|
| `aa_baseline` | Both unprompted | Both unprompted |
| `aa_o_enjoying` | Both prompted (o-enjoying) | Both unprompted |
| `aa_o_avoiding` | Both prompted (o-avoiding) | Both unprompted |

### Output

Each experiment writes to `scratch/runs/o_frequency/{name}_{timestamp}/`:
- Canonical conversation data (samples, events, exports)
- `per_message_metrics.jsonl` — per-message count_o scores with metadata
- `experiment_metadata.json` — git hash, script name, CLI args, timestamp

With `--hf-repo`, the entire run directory is uploaded to HuggingFace.

### Architecture

The experiment script is thin — it defines the experiment matrix and evaluation config. Generic infrastructure lives in `scripts/rollout_generation/experiment_utils.py` (CLI args, config builders, phase runner, HF upload).
