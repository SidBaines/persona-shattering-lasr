# LLM Judge Calibration

Infrastructure for calibrating and running LLM-as-judge personality and coherence scoring.

## Quick start

### Score a response

```python
from src_dev.persona_metrics.config import judge_config
from src_dev.persona_metrics.registry import get_persona_metric

# Get a judge metric (OCEAN or coherence)
metric = get_persona_metric("agreeableness_v2", judge_config=judge_config("gemini_flash"))
result = await metric.evaluate_async("I'd rather just help than argue about it.", "How do you handle conflict?")
# → {"agreeableness_v2.score": 3, "agreeableness_v2.reasoning": "..."}

# Coherence
metric = get_persona_metric("coherence_v2", judge_config=judge_config("haiku"))
result = await metric.evaluate_async("Inflation is when...", "What causes inflation?")
# → {"coherence_v2.score": 8, "coherence_v2.reasoning": "..."}
```

### Available judges

| Registry name | Scale | Definition |
|---|---|---|
| `agreeableness_v2` | -4..+4 | `src_dev/persona_metrics/metrics/ocean_v2.py` |
| `conscientiousness_v2` | -4..+4 | same |
| `extraversion_v2` | -4..+4 | same |
| `neuroticism_v2` | -4..+4 | same |
| `openness_v2` | -4..+4 | same |
| `coherence_v2` | 0..10 | `src_dev/persona_metrics/metrics/coherence.py` |

Legacy aliases (`"agreeableness"`, `"coherence"`, `"better_coherence_judge"`) resolve to the v2 classes.

### Recommended judge models

Defined in `src_dev/persona_metrics/config.py` → `JUDGE_PANEL`:

```python
from src_dev.persona_metrics.config import judge_config

cfg = judge_config("gemini_flash")   # Default — cheapest, fast, good quality
cfg = judge_config("haiku")          # Best quality + throughput balance
cfg = judge_config("kimi_k2")        # Best quality, rate-limited (50 rpm)
cfg = judge_config("deepseek_v3")    # Cheap fallback
```

All models go through OpenRouter. Set `OPENROUTER_API_KEY` in `.env`.

### Scoring a batch (e.g. distillation data or sweep responses)

```python
"""Score distillation responses on all OCEAN traits + coherence."""
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from src_dev.persona_metrics.config import judge_config
from src_dev.persona_metrics.registry import get_persona_metric

load_dotenv()

TRAITS = ["agreeableness_v2", "neuroticism_v2", "openness_v2",
          "conscientiousness_v2", "extraversion_v2", "coherence_v2"]
JUDGE = judge_config("gemini_flash")  # or "haiku" for higher quality


async def score_file(path: Path) -> list[dict]:
    data = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    responses = [d["response"] for d in data]
    questions = [d["question"] for d in data]

    for trait in TRAITS:
        metric = get_persona_metric(trait, judge_config=JUDGE)
        results = await metric.evaluate_batch_async(responses, questions)
        for d, r in zip(data, results):
            d.update(r)  # adds e.g. "agreeableness_v2.score", "agreeableness_v2.reasoning"

    return data


scored = asyncio.run(score_file(Path("scratch/my_responses.jsonl")))
Path("scratch/my_responses_scored.jsonl").write_text(
    "\n".join(json.dumps(d) for d in scored)
)
```

### Using judges in a cross-trait bleed check

```python
"""Score distillation data for cross-trait bleed (OCT pipeline style)."""
from src_dev.persona_metrics.config import judge_config

# Define a panel — gemini for speed, kimi for quality on target trait
JUDGE_CONFIGS = {
    "gemini_flash": judge_config("gemini_flash"),
    "kimi_k2": judge_config("kimi_k2"),
}

# Use with scripts_dev/oct_pipeline/judge_distillation.py
# or pass directly to any metric:
metric = get_persona_metric("agreeableness_v2", judge_config=JUDGE_CONFIGS["kimi_k2"])
```

Cost estimates per calibration run (33 items × 3 repeats = 99 calls):

| Model | Input $/M | Output $/M | ~Cost/run |
|---|---|---|---|
| Gemini Flash | $0.10 | $0.40 | $0.02 |
| Llama 4 Scout | $0.15 | $0.42 | $0.03 |
| DeepSeek V3 | $0.30 | $0.88 | $0.07 |
| Kimi K2 | $0.60 | $2.40 | $0.14 |
| Haiku 3.5 | $0.80 | $4.00 | $0.20 |

Full 6-trait calibration (all OCEAN + coherence) ≈ 6× above. Prices as of April 2026.

## Calibration

### Golden datasets

Hand-labeled calibration items in `data/judge_calibration/`:

```
data/judge_calibration/
  agreeableness.jsonl      # 36 items, -4..+4
  conscientiousness.jsonl  # 36 items, -4..+4
  extraversion.jsonl       # 36 items, -4..+4
  neuroticism.jsonl        # 36 items, -4..+4
  openness.jsonl           # 36 items, -4..+4
  coherence.jsonl          # 33 items, 0..10
```

Each item: `{id, trait, question, response, gold_score, notes}`.

### Run calibration

```bash
# Score one trait with default judge (Gemini Flash, 3 repeats at temp=0.7)
uv run python scripts_dev/persona_metrics/llm_judge/golden_calibration.py score \
    --trait coherence

# All traits
uv run python scripts_dev/persona_metrics/llm_judge/golden_calibration.py score

# Specific model
uv run python scripts_dev/persona_metrics/llm_judge/golden_calibration.py score \
    --trait coherence --model anthropic/claude-3.5-haiku

# Kimi (use lower concurrency to avoid 429s)
uv run python scripts_dev/persona_metrics/llm_judge/golden_calibration.py score \
    --trait coherence --model moonshotai/kimi-k2 --max-concurrent 3

# Resume a partial run
uv run python scripts_dev/persona_metrics/llm_judge/golden_calibration.py score \
    --trait coherence --model moonshotai/kimi-k2 \
    --run-dir scratch/golden_calibration/<run_dir>

# Compare all completed runs
uv run python scripts_dev/persona_metrics/llm_judge/golden_calibration.py compare

# Upload to HuggingFace
uv run python scripts_dev/persona_metrics/llm_judge/golden_calibration.py upload \
    --repo persona-shattering-lasr/monorepo
```

Output goes to `scratch/golden_calibration/<model>__r<repeats>__<timestamp>/`.

### Human annotation

Generate mobile-friendly HTML annotation interfaces:

```bash
# All traits for one rater
uv run python scripts_dev/persona_metrics/llm_judge/generate_annotation_html.py \
    --rater alice

# Single trait
uv run python scripts_dev/persona_metrics/llm_judge/generate_annotation_html.py \
    --trait coherence --rater alice
```

Output: `scratch/annotation_html/annotate_<trait>_<rater>.html`

Open in a browser, score all items, download JSON. Items are randomised (seed=42)
so raters don't see gold-score ordering. Progress is saved in localStorage.

## File map

```
src_dev/
  common/
    persona_definitions.py     # OCEAN trait definitions (facets, examples)
    coherence_definition.py    # Coherence dimension definitions (rubric, failure modes)
  persona_metrics/
    config.py                  # JudgeLLMConfig, JUDGE_PANEL, judge_config()
    metrics/
      ocean_v2.py              # 5 OCEAN judge classes (built from persona_definitions)
      coherence.py             # CoherenceV2Evaluation (built from coherence_definition)
      llm_judge_base.py        # LLMJudgeMetric base class

data/judge_calibration/        # Golden datasets (hand-labeled, checked in)

scripts_dev/persona_metrics/llm_judge/
  golden_calibration.py        # Score goldens, compute agreement, compare models
  generate_annotation_html.py  # Generate human annotation HTML
  plot_judge_calibration.py    # Publication-quality plots from HF or local data
  ocean_judge_calibration.py   # Two-stage OCEAN calibration (generate + judge)

scratch/golden_calibration/    # Calibration run outputs (gitignored)
scratch/annotation_html/       # Generated HTML files (gitignored)
```

## Known issues

- **Kimi K2 rate limits:** OpenRouter imposes 50 rpm on `moonshotai/kimi-k2`.
  Use `--max-concurrent 3` and expect retries with 429s. The built-in exponential
  backoff handles this but runs are slower.

- **GPT-5 Mini (retired):** OpenRouter routes this through Azure, which returns
  403 content-policy blocks on personality assessment prompts. Do not use.
