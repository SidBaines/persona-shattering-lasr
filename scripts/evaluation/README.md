# Evaluation

Run evaluations on datasets at any pipeline stage. Evaluations compute metrics
on model responses and can be used after inference, after editing, during
training, or on ad-hoc model+dataset combinations.

## CLI Usage

```bash
# Count 'o' characters in responses
uv run python -m scripts.evaluation \
  --evaluations count_o \
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
  --evaluations count_o coherence \
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
    evaluations=["count_o"],
    response_column="response",
    output_path=Path("scratch/eval_results.jsonl"),
)
dataset, result = run_evaluation(config, dataset=my_dataset)

# LLM-as-judge evaluation
config = EvaluationConfig(
    evaluations=[
        "count_o",
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
# {"count_o.count.mean": 3.5, "coherence.score.mean": 78.2, ...}

```

## Available Evaluations

- **`count_o`**: Counts occurrences of the letter 'o' (case-insensitive). Returns count
  and density. No external dependencies.
- **`verb_count`**: Counts verb tokens using spaCy POS tagging. Returns count
  and density. Requires `spacy` and the `en_core_web_sm` model.
- **`coherence`**: Uses an LLM judge to rate response coherence from 0-100.
  Returns score and reasoning. Requires API key for the judge provider.
- **`lowercase_density`**: Counts lowercase letters. Returns count and density.
- **`punctuation_density`**: Counts punctuation characters. Returns count and density.

## Persona Registry

Each persona maps to an evaluation and an editing prompt template, registered
in `scripts.common.persona_metrics`. The `--persona` flag on CLI tools is a
convenience that resolves to the right evaluation and prompt template.

### Built-in personas

| Persona | Evaluation | Editing Prompt Template |
|---------|------------|------------------------|
| `o_avoiding` | `count_o` | `default_persona_shatter` |
| `verbs_avoiding` | `verb_count` | `verbs_persona_shatter` |

### Usage

The `--persona` flag on the evaluation and editing CLIs resolves the persona
to its corresponding evaluation:

```bash
# These are equivalent:
uv run python -m scripts.evaluation \
  --evaluations count_o \
  --dataset-path scratch/inference_output.jsonl

uv run python -m scripts.evaluation \
  --persona o_avoiding \
  --dataset-path scratch/inference_output.jsonl

# Editing with a specific persona (sets prompt template + quality eval)
uv run python -m scripts.editing \
  --persona verbs_avoiding \
  --input-path scratch/inference_output.jsonl \
  --output-path scratch/edited_dataset.jsonl
```

Note: Training is persona-agnostic — it trains on whatever edited data it receives.

### Python usage

```python
from scripts.common.persona_metrics import get_persona_evaluation, PERSONA_EVALUATIONS

# List available personas
print(list(PERSONA_EVALUATIONS.keys()))  # ["o_avoiding", "verbs_avoiding"]

# Resolve persona to evaluation name
eval_name = get_persona_evaluation("o_avoiding")  # "count_o"
```

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
