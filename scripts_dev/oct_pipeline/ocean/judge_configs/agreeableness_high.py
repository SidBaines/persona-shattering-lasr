"""Cross-trait judge config for A+ distillation data.

Scores teacher and student responses on all 5 OCEAN dimensions to check:
1. Target trait (agreeableness): teacher should score high, student near neutral
2. Other traits: check for cross-trait bleed from high-A constitution
"""

from src_dev.persona_metrics.config import JUDGE_PANEL, JudgeLLMConfig

# Path to the distillation JSONL file
DATA_PATH = "scratch/hf_download_a_plus/fine_tuning/llama-3.1-8b-it/ocean/agreeableness/amplifier/v1/data/distillation/agreeableness_high.jsonl"

# Which traits to judge — "all" runs all 5 OCEAN traits
JUDGE_NAME = "all"

# Output directory for scored results
OUTPUT_DIR = "scratch/judge_runs/agreeableness_high_distillation_v1"

# Column name for student responses in the distillation JSONL
STUDENT_COLUMN = "llama-3.1-8b-it"

# Judge LLM panel — uses canonical 3-judge panel
JUDGE_CONFIGS = dict(JUDGE_PANEL)

# Default single judge
JUDGE_CONFIG = JudgeLLMConfig()
