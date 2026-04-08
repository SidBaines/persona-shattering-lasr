"""Cross-trait judge config for A+ distillation data.

Scores teacher and student responses on all 5 OCEAN dimensions to check:
1. Target trait (agreeableness): teacher should score high, student near neutral
2. Other traits: check for cross-trait bleed from high-A constitution
"""

from src_dev.persona_metrics.config import judge_config

# Path to the distillation JSONL file
DATA_PATH = "scratch/hf_download_a_plus/fine_tuning/llama-3.1-8b-it/ocean/agreeableness/amplifier/v1/data/distillation/agreeableness_high.jsonl"

# Which traits to judge — "all" runs all 5 OCEAN traits
JUDGE_NAME = "all"

# Output directory for scored results
OUTPUT_DIR = "scratch/judge_runs/agreeableness_high_distillation_v1"

# Column name for student responses in the distillation JSONL
STUDENT_COLUMN = "llama-3.1-8b-it"

# Judge LLM panel — uses canonical panel from src_dev.persona_metrics.config
JUDGE_CONFIGS = {
    "kimi_k2": judge_config("kimi_k2"),
    "gemini_flash": judge_config("gemini_flash"),
}

# Default single judge
JUDGE_CONFIG = JUDGE_CONFIGS["gemini_flash"]
