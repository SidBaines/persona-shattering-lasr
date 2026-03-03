# Conscientious Dataset Comparative Judge

This experiment mirrors `scripts/experiments/conscientious-dataset-judge/`, but uses the comparative conscientiousness judge instead of scoring each response independently.

That means each question's sampled responses are judged together, then written back out as per-response scores for filtering and review.

## End-to-end workflow

From "no local dataset yet" to "trained model", the intended flow is:

1. Generate sampled responses from the source dataset.
2. Judge each question's sampled responses comparatively for conscientiousness.
3. Review the judged output in the TUI if needed.
4. Filter to questions where the high-vs-low score gap is at least `N`.
5. Keep the lower-scoring extreme response for each qualifying question.
6. Train a LoRA on that filtered dataset with the standard training wrapper.

If you already have `inference_responses.jsonl`, skip to step 2. If you already have `judged_responses.jsonl`, skip to step 4.

## Prerequisites

- Put API keys in `.env` as needed:
  - `OPENAI_API_KEY` for OpenAI judging
  - `ANTHROPIC_API_KEY` for Anthropic judging
- Use `uv run python ...` so the project environment and dependencies are available.

## Files

The experiment script is:

- `scripts/experiments/conscientious-dataset-comparative-judge/build_dataset.py`

Each run directory contains:

```text
<run-dir>/
├── run_config.json
├── inference_responses.jsonl
├── judged_responses.jsonl
├── gap_filtered_training_candidates.jsonl
├── gap_filter_summary.json
├── training_candidates.jsonl
└── summary.json
```

## Default run

```bash
uv run python scripts/experiments/conscientious-dataset-comparative-judge/build_dataset.py \
    --run-dir scratch/conscientious_comparative_judge_run
```

Defaults:

- dataset: `liweijiang/infinite-chats-taxonomy`
- question field: `lm_judge_annotation.revised_query`
- questions: `100`
- responses per question: `5`
- seed: `42`
- inference temperature: `1.2`
- min comparative conscientiousness score kept for training: `4`

If you want the larger original setting explicitly:

```bash
uv run python scripts/experiments/conscientious-dataset-comparative-judge/build_dataset.py \
    --run-dir scratch/conscientious_comparative_judge_run \
    --max-samples 100 \
    --responses-per-question 5
```

## Run phases separately

Inference only:

```bash
uv run python scripts/experiments/conscientious-dataset-comparative-judge/build_dataset.py \
    --run-dir scratch/conscientious_comparative_judge_run \
    --phase inference
```

Judge only:

```bash
uv run python scripts/experiments/conscientious-dataset-comparative-judge/build_dataset.py \
    --run-dir scratch/conscientious_comparative_judge_run \
    --phase judge
```

Behavior:

- inference is skipped automatically if `inference_responses.jsonl` already exists
- judging resumes at full question-group boundaries
- if a partial judged tail exists, it is trimmed back to the last complete question-group
- `training_candidates.jsonl` is rebuilt from the judged file each time the judge phase finishes

## Review in the TUI

```bash
uv run python scripts/jsonl_tui/cli.py scratch/conscientious_comparative_judge_run/judged_responses.jsonl \
    --display-fields conscientiousness_comparative_score question response conscientiousness_comparative_reasoning
```

Navigation:

- `Up` / `Down`: previous or next question
- `Left` / `Right`: previous or next response for the current question
- `j` / `k`: scroll within the current response
- `q`: quit

This is the easiest point to sanity-check whether the comparative judge is actually separating stronger and weaker responses within each question group.

## Useful overrides

Use Anthropic for the judge:

```bash
uv run python scripts/experiments/conscientious-dataset-comparative-judge/build_dataset.py \
    --run-dir scratch/conscientious_comparative_judge_run \
    --judge-provider anthropic \
    --judge-model claude-haiku-4-5-20251001
```

Keep only stronger positives:

```bash
uv run python scripts/experiments/conscientious-dataset-comparative-judge/build_dataset.py \
    --run-dir scratch/conscientious_comparative_judge_run \
    --phase judge \
    --min-score 6
```

## Filter for large score gaps

This keeps only questions where the comparative judge produced a large spread across the sampled responses.

For each qualifying question:

- compute `max(score) - min(score)`
- keep the question only if that gap is at least `N`
- export the lower-scoring of the two extreme responses for training

Default `N` is `3`.

```bash
uv run python scripts/experiments/conscientious-dataset-comparative-judge/filter_gap_training_data.py \
    --run-dir scratch/conscientious_comparative_judge_run
```

Use a different threshold:

```bash
uv run python scripts/experiments/conscientious-dataset-comparative-judge/filter_gap_training_data.py \
    --run-dir scratch/conscientious_comparative_judge_run \
    --min-gap 5
```

This writes:

- `gap_filtered_training_candidates.jsonl`
- `gap_filter_summary.json`

The exported dataset contains one row per qualifying question:

- `question`
- `response`
- `conscientiousness_comparative_score`
- `selection_gap`
- contrast metadata for the higher-scoring paired response

## Train on the gap-filtered data

The filtered dataset already has the right `question` and `response` columns for the existing training wrapper.

Example:

```bash
uv run python scripts/experiments/persona_pipelines/persona_training.py \
    --dataset-path scratch/conscientious_comparative_judge_run/gap_filtered_training_candidates.jsonl \
    --user-column question \
    --assistant-column response \
    --group-column question \
    --evaluations conscientiousness \
    --run-id conscientious-gap-train \
    --skip-hf-upload
```

Notes:

- `--group-column question` prevents train/val leakage across repeated prompts
- `--evaluations conscientiousness` uses the standard per-response judge during training-time eval
- if you want a different base model, add `--hf-model ...`
- if you want uploads enabled, remove `--skip-hf-upload`

## One concrete end-to-end example

Generate and judge:

```bash
uv run python scripts/experiments/conscientious-dataset-comparative-judge/build_dataset.py \
    --run-dir scratch/conscientious_comparative_judge_run \
    --max-samples 100 \
    --responses-per-question 5
```

Review:

```bash
uv run python scripts/jsonl_tui/cli.py scratch/conscientious_comparative_judge_run/judged_responses.jsonl \
    --display-fields conscientiousness_comparative_score question response conscientiousness_comparative_reasoning
```

Filter for large gaps:

```bash
uv run python scripts/experiments/conscientious-dataset-comparative-judge/filter_gap_training_data.py \
    --run-dir scratch/conscientious_comparative_judge_run \
    --min-gap 3
```

Train:

```bash
uv run python scripts/experiments/persona_pipelines/persona_training.py \
    --dataset-path scratch/conscientious_comparative_judge_run/gap_filtered_training_candidates.jsonl \
    --user-column question \
    --assistant-column response \
    --group-column question \
    --evaluations conscientiousness \
    --run-id conscientious-gap-train \
    --skip-hf-upload
```

## Help

```bash
uv run python scripts/experiments/conscientious-dataset-comparative-judge/build_dataset.py --help
```
