# Evaluation

Run evaluations on datasets at any pipeline stage. Evaluations compute metrics
on model responses and can be used after inference, after editing, during
training, or on ad-hoc model+dataset combinations.

## CLI Usage

```bash
# Level of persona evaluation on inference output
uv run python -m scripts.evaluation \
  --evaluations level_of_persona \
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
  --evaluations level_of_persona coherence \
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
    evaluations=["level_of_persona"],
    response_column="response",
    output_path=Path("scratch/eval_results.jsonl"),
)
dataset, result = run_evaluation(config, dataset=my_dataset)

# LLM-as-judge evaluation
config = EvaluationConfig(
    evaluations=[
        "level_of_persona",
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
# {"level_of_persona.count.mean": 3.5, "coherence.score.mean": 78.2, ...}

```

## Available Evaluations

- **`level_of_persona`**: Measures persona adherence level in responses. Returns count
  and density. The concrete measurement function is determined by the active persona
  (see [Persona Registry](#persona-registry) below). No external dependencies.
- **`coherence`**: Uses an LLM judge to rate response coherence from 0-100.
  Returns score and reasoning. Requires API key for the judge provider.

## Persona Registry

The `level_of_persona` evaluation delegates its measurement to a **persona metric**
registered in `scripts.common.persona_metrics`. Each metric function returns `count`
and `density` values that quantify how strongly a persona trait manifests in text.

### Built-in personas

| Name | Description | Measurement |
|------|-------------|-------------|
| `o_avoiding` | Avoids the letter "o" | Count of "o" characters (case-insensitive) |
| `verbs_avoiding` | Avoids verbs | Verb token count via spaCy POS tagging |

Additional personas (e.g., `verb_avoiding`, `formal_tone`) will be added as needed.

### Usage

The persona is selected via `--persona` on the evaluation and editing CLIs.
Both modules resolve measurement from the same shared registry:

```bash
# Evaluation with a specific persona
uv run python -m scripts.evaluation \
  --persona o_avoiding \
  --evaluations level_of_persona \
  --dataset-path scratch/inference_output.jsonl

# Editing with quality metrics using a specific persona
uv run python -m scripts.editing \
  --persona o_avoiding \
  --input-path scratch/inference_output.jsonl \
  --output-path scratch/edited_dataset.jsonl
```

Note: Training is persona-agnostic — it trains on whatever edited data it receives.
The `level_of_persona` evaluation during training still needs a `--persona` flag
on the evaluation config, but training itself doesn't care which persona was used.

### Python usage

```python
from scripts.common.persona_metrics import get_persona_metric, PERSONA_METRICS

# List available personas
print(list(PERSONA_METRICS.keys()))  # ["o_avoiding", ...]

# Get a persona metric function and evaluate text
metric_fn = get_persona_metric("o_avoiding")
result = metric_fn("Hello world")  # {"count": ..., "density": ...}
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
