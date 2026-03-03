# JSONL TUI

## How to run

To run the tui on a jsonl file, from the main directory:
`uv run python scripts/jsonl_tui/cli.py scratch/my_exp/my_file.jsonl`

## Variant comparison mode

Pass `--variant-fields` to compare multiple text variants side by side (Left/Right to cycle):

```bash
uv run python scripts/jsonl_tui/cli.py scratch/my_exp/compare.jsonl \
    --variant-fields original neutral_paraphrase_control o-
```

## Meta fields (header badges)

Pass `--meta-fields` to display numeric or string values as badges in the header bar.
These are shown alongside the question/variant counters and are not treated as variant panels.
Useful for scores or other per-record metadata:

```bash
uv run python scripts/jsonl_tui/cli.py scratch/my_exp/compare.jsonl \
    --variant-fields original o- \
    --meta-fields original_openness_score
```

The header will show e.g. `[original_openness_score: +7]` for each question.

### Per-tab score convention (variant mode)

In variant-fields mode, meta fields are automatically filtered so that each tab
only shows the scores relevant to the currently displayed variant.

**Convention:** name your score fields `<variant>_<metric>` (e.g.
`o-_openness_score`, `o-_coherence_score`).  A field is shown on the
`<variant>` tab if its name starts with `<variant>_`.  Fields that don't start
with any known variant name are treated as **global** and shown on every tab.

Example — pass all score fields at once and let the TUI sort them out:

```bash
uv run python scripts/jsonl_tui/cli.py scratch/my_exp/compare.jsonl \
    --variant-fields original neutral_paraphrase_control o- \
    --meta-fields \
        original_openness_score \
        neutral_paraphrase_control_openness_score \
        neutral_paraphrase_control_coherence_score \
        o-_openness_score \
        o-_coherence_score
```

On the `original` tab you will see `original_openness_score`; on the `o-` tab
you will see `o-_openness_score` and `o-_coherence_score`; and so on.

The `openness_iteration.py` pipeline follows this convention when it writes
`compare.jsonl`, so you can copy the `--meta-fields` list straight from the
"View results" hint it prints at the end.