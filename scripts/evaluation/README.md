# Evaluation

Run evaluations on datasets at any pipeline stage. Evaluations compute metrics
on model responses and can be used after inference, after editing, during
training, or on ad-hoc model+dataset combinations.

## CLI Usage

```bash
# Count verbs evaluation on inference output
uv run python -m scripts.evaluation \
  --evaluations count_verbs \
  --dataset-path scratch/inference_output.jsonl \
  --output-path scratch/eval_results.jsonl

# Coherence evaluation using LLM judge
uv run python -m scripts.evaluation \
  --evaluations coherence \
  --judge-provider openai \
  --judge-model gpt-4o-mini \
  --dataset-path scratch/inference_output.jsonl \
  --output-path scratch/eval_results.jsonl

# Multiple evaluations at once
uv run python -m scripts.evaluation \
  --evaluations count_verbs coherence \
  --dataset-path scratch/edited_dataset.jsonl \
  --response-column edited_response \
  --output-path scratch/eval_results.jsonl
```

## Python Usage

```python
from pathlib import Path
from scripts.evaluation import run_evaluation, EvaluationConfig, EvaluationSpec, JudgeLLMConfig

# Simple evaluation (no LLM needed)
config = EvaluationConfig(
    evaluations=["count_verbs"],
    response_column="response",
    output_path=Path("scratch/eval_results.jsonl"),
)
dataset, result = run_evaluation(config, dataset=my_dataset)

# LLM-as-judge evaluation
config = EvaluationConfig(
    evaluations=[
        "count_verbs",
        "coherence",
    ],
    response_column="edited_response",
    judge=JudgeLLMConfig(
        provider="openai",
        model="gpt-4o-mini",
        max_concurrent=20,
    ),
    output_path=Path("scratch/eval_results.jsonl"),
)
dataset, result = run_evaluation(config, dataset=edited_dataset)

# Access aggregate results
print(result.aggregates)
# {"count_verbs.count.mean": 3.5, "coherence.score.mean": 78.2, ...}
```

## Available Evaluations

- **`count_verbs`**: Counts verbs in responses using spacy POS tagging. Returns count
  and density (percentage of tokens). Requires spacy with en_core_web_sm model.
- **`coherence`**: Uses an LLM judge to rate response coherence from 0-100.
  Returns score and reasoning. Requires API key for the judge provider.

### Custom coherence prompt

You can override the coherence judge prompt and examples via `EvaluationSpec.params`:

```python
from scripts.evaluation import EvaluationConfig, EvaluationSpec

custom_template = (
    "Score coherence 0-100.\n"
    "{examples_text}\n"
    "Question: {question_text}\n"
    "Response: {response}\n"
    'Reply as JSON: {{"score": <int>, "reasoning": "<brief>"}}'
)

custom_examples = [
    {
        "question": "What is photosynthesis?",
        "response": "Photosynthesis is how plants make food using sunlight.",
        "score": 90,
        "reasoning": "Direct, on-topic, and logically organized.",
    },
]

config = EvaluationConfig(
    evaluations=[
        EvaluationSpec(
            name="coherence",
            params={
                "prompt_template": custom_template,
                "examples": custom_examples,
            },
        ),
    ],
)
```

## Custom Evaluations

```python
from scripts.evaluation import Evaluation, register_evaluation

class MyEvaluation(Evaluation):
    @property
    def name(self) -> str:
        return "my_eval"

    def evaluate(self, response: str, question: str | None = None) -> dict:
        return {f"{self.name}.score": compute_score(response)}

register_evaluation("my_eval", MyEvaluation)
```
