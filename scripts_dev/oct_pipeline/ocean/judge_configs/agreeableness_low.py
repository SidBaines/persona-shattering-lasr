"""Cross-trait judge config for A-minus distillation data.

Scores teacher and student responses on all 5 OCEAN dimensions to check:
1. Target trait (agreeableness): teacher should score low, student near neutral
2. Other traits: teacher should NOT deviate significantly from student,
   unless the teacher model (GLM) has inherent baseline differences
"""

from src_dev.persona_metrics.config import JUDGE_PANEL, JudgeLLMConfig

# Path to the distillation JSONL file
DATA_PATH = "scratch/hf_download_v2/fine_tuning/llama-3.1-8b-it/ocean/agreeableness/suppressor/v2/data/distillation/agreeableness_low.jsonl"

# Which traits to judge — "all" runs all 5 OCEAN traits
JUDGE_NAME = "all"

# Output directory for scored results
OUTPUT_DIR = "scratch/judge_runs/agreeableness_low_distillation_v2_shuffled"

# Column name for student responses in the distillation JSONL
STUDENT_COLUMN = "llama-3.1-8b-it"

# Judge LLM panel — uses canonical 3-judge panel
JUDGE_CONFIGS = dict(JUDGE_PANEL)

# Default single judge (used if runner doesn't support panel)
JUDGE_CONFIG = JudgeLLMConfig()
