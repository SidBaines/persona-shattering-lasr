# Conscientious Dataset Judge

This experiment generates many candidate responses, scores them with the existing conscientiousness LLM judge, and exports the high-scoring rows as training candidates.

It is designed for the workflow you described:

- sample `100` questions from `liweijiang/infinite-chats-taxonomy`
- shuffle with a fixed random seed
- generate `5` high-temperature responses per question
- judge every response for conscientiousness
- keep only the higher-scoring rows for training

## Files

The experiment script is:

- `scripts/experiments/conscientious-dataset-judge/build_dataset.py`

Each run directory contains:

```text
<run-dir>/
├── run_config.json
├── inference_responses.jsonl
├── judged_responses.jsonl
├── training_candidates.jsonl
└── summary.json
```

## Default run

```bash
uv run python scripts/experiments/conscientious-dataset-judge/build_dataset.py \
    --run-dir scratch/conscientious_judge_run
```

Defaults:

- dataset: `liweijiang/infinite-chats-taxonomy`
- question field: `lm_judge_annotation.revised_query`
- questions: `100`
- responses per question: `5`
- seed: `42`
- inference temperature: `1.2`
- min conscientiousness score kept for training: `4`

## Run phases separately

Inference only:

```bash
uv run python scripts/experiments/conscientious-dataset-judge/build_dataset.py \
    --run-dir scratch/conscientious_judge_run \
    --phase inference
```

Judge only:

```bash
uv run python scripts/experiments/conscientious-dataset-judge/build_dataset.py \
    --run-dir scratch/conscientious_judge_run \
    --phase judge
```

Behavior:

- inference is skipped automatically if `inference_responses.jsonl` already exists
- judging resumes automatically if `judged_responses.jsonl` is partially complete
- `training_candidates.jsonl` is rebuilt from the judged file each time the judge phase finishes

## Review in the TUI

Use the shared JSONL TUI in grouped-response mode:

```bash
uv run python scripts/jsonl_tui/cli.py scratch/conscientious_judge_run/judged_responses.jsonl \
    --display-fields conscientiousness_score question response conscientiousness_reasoning
```

Navigation:

- `Up` / `Down`: previous or next question
- `Left` / `Right`: previous or next response for the current question
- `j` / `k`: scroll within the current response
- `q`: quit

This gives the view you wanted:

- question
- current sampled response
- conscientiousness score
- judge reasoning

## Useful overrides

Use a different inference model:

```bash
uv run python scripts/experiments/conscientious-dataset-judge/build_dataset.py \
    --run-dir scratch/conscientious_qwen \
    --inference-model Qwen/Qwen2.5-7B-Instruct
```

Use a different judge model:

```bash
uv run python scripts/experiments/conscientious-dataset-judge/build_dataset.py \
    --run-dir scratch/conscientious_judge_run \
    --judge-provider anthropic \
    --judge-model claude-sonnet-4-20250514
```

Keep only stronger positives:

```bash
uv run python scripts/experiments/conscientious-dataset-judge/build_dataset.py \
    --run-dir scratch/conscientious_judge_run \
    --phase judge \
    --min-score 6
```

## Help

```bash
uv run python scripts/experiments/conscientious-dataset-judge/build_dataset.py --help
```
