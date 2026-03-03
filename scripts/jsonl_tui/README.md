# JSONL TUI

## How to run

To run the tui on a jsonl file, from the main directory:
`uv run python scripts/jsonl_tui/cli.py scratch/my_exp/my_file.jsonl`

For grouped response browsing with a cleaner prose view instead of raw JSON:
`uv run python scripts/jsonl_tui/cli.py scratch/my_exp/judged.jsonl --display-fields question response conscientiousness_score conscientiousness_reasoning`

For side-by-side variant comparison:
`uv run python scripts/jsonl_tui/cli.py scratch/my_exp/compare.jsonl --variant-fields original c+v1 c-v1`
