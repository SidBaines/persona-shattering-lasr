# Evaluation

Metrics for measuring persona alignment in model responses.

## Overview

This module provides metrics to evaluate how well model responses exhibit target persona traits. Supports both simple code-based metrics and LLM-as-judge approaches.

## Usage

```python
from src.evaluation import get_metric

# Simple character counting
metric = get_metric("count_char")
score = metric.compute(
    response="Hello world",
    config={"char": "o", "normalize": True}
)

# Batch evaluation with aggregation
scores = metric.compute_batch(responses, config)
summary = metric.aggregate(scores)
print(f"Mean: {summary['mean']:.3f}, Std: {summary['std']:.3f}")
```

## Available Metrics

| Type | Description | Status |
|------|-------------|--------|
| `count_char` | Count character occurrences | STUB |
| `llm_judge` | LLM-as-judge evaluation | STUB |

## Adding a New Metric

1. Create a new file in `metrics/` (e.g., `semantic.py`)
2. Implement the `Metric` interface from `base.py`
3. Register in `metrics/__init__.py`:

```python
from .semantic import SemanticSimilarityMetric

METRICS = {
    "count_char": CharCountMetric,
    "llm_judge": LLMJudgeMetric,
    "semantic": SemanticSimilarityMetric,  # Add here
}
```

## Configuration

In YAML config:

```yaml
evaluation:
  metrics:
    - type: count_char
      char: o
      normalize: true

    - type: llm_judge
      provider: anthropic
      model: claude-sonnet-4-20250514
      prompt_template: |
        Rate how much this response uses the letter 'O' (1-5):
        {response}
      scale: 5
```

## Before Implementing

**REMINDER:** Check what exists in `src/` before implementing in `scripts/`. Use utilities from `src/` when working in `scripts/`.
