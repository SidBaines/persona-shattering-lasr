# Neuroticism LoRA Scale Sweep

Scale sweep comparing 4 neuroticism adapter training approaches (sft, dpo, soup, old)
across 17 scale points (−2.0 to +2.0, step 0.25) on 100 assistant-axis questions.

## Directory structure

```
neuroticism/
  inference/     — generate rollouts (neurotic_lora_sweep.py)
  judges/        — run judge panels (coherence, neuroticism, gemini consistency)
  analysis/      — aggregate scores, plots, and reports
```

## Running the sweep

```bash
# All 4 adapters sequentially (unattended)
uv run python -m scripts_dev.rollout_experiments.neuroticism.inference.neurotic_lora_sweep

# Single adapter
uv run python -m scripts_dev.rollout_experiments.neuroticism.inference.neurotic_lora_sweep sft

# Subset
uv run python -m scripts_dev.rollout_experiments.neuroticism.inference.neurotic_lora_sweep dpo soup old
```

Run in tmux to survive disconnects:
```bash
tmux new -s neurotic_sweep
uv run python -m scripts_dev.rollout_experiments.neuroticism.inference.neurotic_lora_sweep
```

## Inspecting rollouts (TUI)

The rollouts.jsonl `messages` field is a dict keyed by rollout index (`"0"`, `"1"`, ...).
Flatten to a list before passing to the TUI:

```bash
python3 -c "
import json, sys
scale = sys.argv[1]   # e.g. scale_+2.00
adapter = sys.argv[2] # e.g. sft
path = f'scratch/runs/fine_tuning/llama-3.1-8B-Instruct/OCEAN/neuroticism/neuroticism_{adapter}/rollouts/neurotic_lora_sweep/{scale}/1turn_astNoSProm___no_prompt/rollouts/rollouts.jsonl'
for line in open(path):
    d = json.loads(line)
    d['conversation'] = d['messages']['0']
    print(json.dumps(d))
" scale_+2.00 sft > /tmp/inspect.jsonl

uv run python -m src_dev.jsonl_tui.cli /tmp/inspect.jsonl --conversation-field conversation
```

Suggested scales to inspect before running judges: `scale_+2.00`, `scale_-2.00`.

## Pipeline

### Step 1 — Convert to judge dataset
```bash
uv run python scripts_dev/persona_metrics/llm_judge/rollout_sweep_to_judge_dataset.py \
    --sweep-dir scratch/runs/fine_tuning/llama-3.1-8B-Instruct/OCEAN/neuroticism/neuroticism_<name>/rollouts/neurotic_lora_sweep \
    --output scratch/judge_datasets/neuroticism_<name>_sweep.jsonl \
    --model meta-llama/Llama-3.1-8B-Instruct
```

### Step 2 — Upload to HF
```bash
huggingface-cli upload --repo-type dataset persona-shattering-lasr/monorepo \
    scratch/judge_datasets/neuroticism_<name>_sweep.jsonl \
    judge_datasets/neuroticism_<name>_sweep.jsonl
```

### Step 3 — Neuroticism judge panel
```bash
uv run python -m scripts_dev.rollout_experiments.neuroticism.judges.neuroticism_judge_sweep_v2 \
    --adapter <name>

# Or via the calibration script directly:
uv run python scripts_dev/persona_metrics/llm_judge/ocean_judge_calibration.py \
    --trait neuroticism --stage judge \
    --dataset scratch/judge_datasets/neuroticism_<name>_sweep.jsonl \
    --upload
```

### Step 3b — Ablation: without score examples
```bash
uv run python scripts_dev/persona_metrics/llm_judge/ocean_judge_calibration.py \
    --trait neuroticism --stage judge \
    --dataset scratch/judge_datasets/neuroticism_<name>_sweep.jsonl \
    --no-examples --upload
```

### Step 4 — Coherence judge
```bash
uv run python -m scripts_dev.rollout_experiments.neuroticism.judges.coherence_lora_sweep_v2

# Or via the calibration script directly:
uv run python scripts_dev/persona_metrics/llm_judge/coherence_calibration.py \
    --stage judge \
    --dataset scratch/judge_datasets/neuroticism_<name>_sweep.jsonl
```

### Step 5 — Analysis
```bash
uv run python -m scripts_dev.rollout_experiments.neuroticism.analysis.coherence_analysis --save --upload
uv run python -m scripts_dev.rollout_experiments.neuroticism.analysis.rerun_variance_analysis --upload
uv run python -m scripts_dev.rollout_experiments.neuroticism.analysis.question_extremes_analysis --save --upload
```

## Adapters

| Name | Path | Rank |
|------|------|------|
| `sft` | `persona-shattering-lasr/monorepo::fine_tuning/.../neuroticism-sft` | r=8 |
| `dpo` | `persona-shattering-lasr/monorepo::fine_tuning/.../neuroticism-dpo` | r=64 (8× larger, ~641MB baked) |
| `soup` | `persona-shattering-lasr/monorepo::fine_tuning/.../neuroticism-persona` | r=8 |
| `old` | `persona-shattering-lasr/20Feb-n-plus::checkpoints/final` | — |

> **Note:** dpo adapter is r=64 vs r=8 for others. Each baked variant is ~641MB,
> totalling ~11GB for 17 scale points. Ensure `baked_adapters_dir` points to a
> volume with sufficient space (currently `/workspace/baked_adapters/`).

## Notes

- `skip_completed=True` skips cells with a local `run_info.json` status=ok — safe to resume after interruption
- Baked adapters are cached at `/workspace/baked_adapters/neuroticism_<name>/` and reused on reruns
- A disk space check runs before baking and raises early if space is insufficient
