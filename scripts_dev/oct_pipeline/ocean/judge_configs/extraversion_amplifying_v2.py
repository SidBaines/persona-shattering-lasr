"""Cross-trait judge config for E+ (amplifying) distillation data v2.

Scores teacher and student responses on all 5 OCEAN dimensions to check:
1. Target trait (extraversion): teacher should score high, student near neutral
2. Other traits: teacher should NOT deviate significantly from student
"""

from src_dev.persona_metrics.config import judge_config

# Path to the distillation JSONL file
DATA_PATH = "scratch/oct_extraversion_amplifying2/data/distillation/extraversion_amplifying_full_v2.jsonl"

# Which traits to judge — "all" runs all 5 OCEAN traits
JUDGE_NAME = "all"

# Output directory for scored results
OUTPUT_DIR = "scratch/judge_runs/extraversion_amplifying_v2_distillation"

# Column name for student responses in the distillation JSONL
STUDENT_COLUMN = "llama-3.1-8b-it"

# Judge LLM panel — uses canonical panel from src_dev.persona_metrics.config
JUDGE_CONFIGS = {
    "kimi_k2": judge_config("kimi_k2"),
    "gemini_flash": judge_config("gemini_flash"),
}

# Default single judge (used if runner doesn't support panel)
JUDGE_CONFIG = JUDGE_CONFIGS["gemini_flash"]
