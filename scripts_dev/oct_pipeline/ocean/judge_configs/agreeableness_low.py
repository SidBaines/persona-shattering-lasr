"""Judge config for A-minus distillation data.

Scores teacher and student responses on the agreeableness dimension.
Expected: teacher scores negative (low-A), student scores near neutral or mildly positive.
"""

from src_dev.persona_metrics.config import JudgeLLMConfig

# Path to the distillation JSONL file
DATA_PATH = "scratch/hf_download_v2/fine_tuning/llama-3.1-8b-it/ocean/agreeableness/suppressor/v2/data/distillation/agreeableness_low.jsonl"

# Which judge to use (must match a key in JUDGE_REGISTRY)
JUDGE_NAME = "agreeableness_v2"

# Output directory for scored results
OUTPUT_DIR = "scratch/judge_runs/agreeableness_low_distillation_v2"

# Column name for student responses in the distillation JSONL
STUDENT_COLUMN = "llama-3.1-8b-it"

# Judge LLM panel — 3 cheap models via OpenRouter
JUDGE_CONFIGS = {
    "gpt_5_mini": JudgeLLMConfig(
        provider="openrouter",
        model="openai/gpt-5-mini",
        temperature=0.0,
        max_concurrent=10,
        max_tokens=512,
    ),
    "kimi_k2": JudgeLLMConfig(
        provider="openrouter",
        model="moonshotai/kimi-k2",
        temperature=0.0,
        max_concurrent=10,
        max_tokens=512,
    ),
    "gemini_flash": JudgeLLMConfig(
        provider="openrouter",
        model="google/gemini-2.0-flash-001",
        temperature=0.0,
        max_concurrent=10,
        max_tokens=512,
    ),
}

# Default single judge (used if runner doesn't support panel)
JUDGE_CONFIG = JUDGE_CONFIGS["gemini_flash"]
