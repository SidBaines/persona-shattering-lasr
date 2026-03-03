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