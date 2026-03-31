"""Judge config for A-minus distillation data.

Scores teacher and student responses on the agreeableness dimension.
Expected: teacher scores negative (low-A), student scores near neutral or mildly positive.
"""

from src_dev.persona_metrics.config import JudgeLLMConfig

# Path to the distillation JSONL file
DATA_PATH = "scratch/hf_download/fine_tuning/llama-3.1-8b-it/ocean/agreeableness/suppressor/v1/data/distillation/agreeableness_low.jsonl"

# Which judge to use (must match a key in JUDGE_REGISTRY)
JUDGE_NAME = "agreeableness_v2"

# Output directory for scored results
OUTPUT_DIR = "scratch/judge_runs/agreeableness_low_distillation"

# Column name for student responses in the distillation JSONL
STUDENT_COLUMN = "llama-3.1-8b-it"

# Judge LLM configuration
JUDGE_CONFIG = JudgeLLMConfig(
    provider="anthropic",
    model="claude-sonnet-4-20250514",
    temperature=0.0,
    max_concurrent=10,
    max_tokens=512,
)
