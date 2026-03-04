# Prompt Iteration Workflow

Tools for rapid iteration on editing prompt templates. The goal is a tight human-in-the-loop cycle: generate a fixed set of base-model responses once, apply candidate prompts to those responses, review the outputs side-by-side in the TUI, refine the prompts, and repeat — without ever re-running inference.

## Scripts

- **`conscientious_iteration.py`** — Inference + multi-variant editing + comparison export for the Conscientiousness (C+/C−) persona prompts. Can be adapted for other traits by changing `--prompts`.

## Iteration workflow

### Step 1 — First run

Generate base-model responses and apply your initial prompt variants:

```bash
uv run python scripts/experiments/prompt-iterations/conscientious_iteration.py \
    --run-dir scratch/c_iter_001 \
    --prompts c+v1 c-v1 neutral_paraphrase_control
```

To run against a committed local prompt set instead of the default HuggingFace dataset:

```bash
uv run python scripts/experiments/prompt-iterations/conscientious_iteration.py \
    --run-dir scratch/c_iter_open_ended_10 \
    --dataset scripts/experiments/prompt-iterations/datasets/conscientious_open_ended_10.jsonl \
    --max-samples 10 \
    --prompts c+v1
```

This creates:

```
scratch/c_iter_001/
├── original_responses.jsonl      # base-model responses (written once, never overwritten)
├── edits/
│   ├── c+v1.jsonl
│   ├── c-v1.jsonl
│   └── neutral_paraphrase_control.jsonl
└── compare.jsonl                  # one row per question, variants as fields
```

The TUI command is printed at the end.

### Step 2 — Review in the TUI

```bash
uv run python scripts/jsonl_tui/cli.py scratch/c_iter_001/compare.jsonl \
    --variant-fields original neutral_paraphrase_control c+v1 c-v1
```

**TUI navigation:**

| Key | Action |
|-----|--------|
| `↑` / `↓` or `n` / `p` | Next / previous question |
| `→` / `←` or `l` / `h` | Next / previous variant |
| `j` / `k` | Scroll down / up within the current variant |
| `PgDn` / `PgUp` | Page down / up |
| `g` / `G` | Jump to first / last question |
| `q` or `ESC` | Quit |

Each view shows the question text and the current variant's response as readable prose. The header shows which variant you're on and how many there are.

### Step 3 — Refine prompts

Edit `scripts/editing/prompts.py` to add new prompt versions, e.g. `c+v2` and `c-v2`.

### Step 4 — Re-run (editing only)

```bash
uv run python scripts/experiments/prompt-iterations/conscientious_iteration.py \
    --run-dir scratch/c_iter_001 \
    --prompts c+v2 c-v2
```

Inference is skipped (original responses already exist). Existing edit variants (`c+v1`, `c-v1`, `neutral_paraphrase_control`) are also skipped. Only the new variants are run.

`compare.jsonl` is rebuilt to include all variants (old and new), so you can compare across iterations:

```bash
uv run python scripts/jsonl_tui/cli.py scratch/c_iter_001/compare.jsonl \
    --variant-fields original neutral_paraphrase_control c+v1 c+v2 c-v1 c-v2
```

### Repeat steps 2–4 until satisfied.

---

## Script options

```
uv run python scripts/experiments/prompt-iterations/conscientious_iteration.py --help
```

| Flag | Default | Description |
|------|---------|-------------|
| `--run-dir PATH` | *(required)* | Directory for run outputs |
| `--prompts TEMPLATE [...]` | *(required)* | Prompt template names from `scripts/editing/prompts.py` |
| `--max-samples N` | `20` | Number of samples to generate via inference |
| `--dataset NAME` | `vicgalle/alpaca-gpt4` | HuggingFace dataset for inference |
| `--inference-model NAME` | `meta-llama/Llama-3.1-8B-Instruct` | Model for inference |
| `--inference-provider STR` | `local` | Inference provider: `local`, `openai`, `anthropic`, `openrouter` |
| `--editing-model NAME` | `claude-sonnet-4-20250514` | Model for editing |
| `--editing-provider STR` | `anthropic` | Editing provider: `anthropic`, `openai` |
| `--overwrite-edits` | off | Force re-run of editing even if output exists |

---

## Adding prompts for a new trait

To adapt this workflow for a different OCEAN trait:

1. Add your new prompt templates to `scripts/editing/prompts.py` (e.g. `a+v1`, `a-v1`).
2. Run the script with `--prompts a+v1 a-v1 neutral_paraphrase_control`.
3. Follow the same iteration loop.

The script is trait-agnostic — `--prompts` accepts any key from the `TEMPLATES` dict.
