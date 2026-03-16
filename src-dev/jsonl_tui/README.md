# JSONL TUI

## How to run

To run the tui on a jsonl file, from the main directory:
`uv run python scripts/jsonl_tui/cli.py scratch/my_exp/my_file.jsonl`

To render multi-turn conversation exports as a chat transcript instead of raw JSON:
`uv run python scripts/jsonl_tui/cli.py scratch/runs/<run_id>/exports/conversation_training.jsonl --conversation-field messages`

In conversation view:
- `Up` / `Down` or `j` / `k` scroll within the current sample
- `Left` / `Right` or `n` / `p` move between samples
