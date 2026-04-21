"""Cross-trait judge config for E+ (amplifying) distillation data v2.

Scores teacher and student responses on all 5 OCEAN dimensions to check:
1. Target trait (extraversion): teacher should score high, student near neutral
2. Other traits: teacher should NOT deviate significantly from student
"""

from src_dev.persona_metrics.config import JUDGE_PANEL, JudgeLLMConfig

# Path to the distillation JSONL file
DATA_PATH = "scratch/oct_extraversion_amplifying2/data/distillation/extraversion_amplifying_full_v2.jsonl"

# Which traits to judge — "all" runs all 5 OCEAN traits
JUDGE_NAME = "all"

# Output directory for scored results
OUTPUT_DIR = "scratch/judge_runs/extraversion_amplifying_v2_distillation"

# Column name for student responses in the distillation JSONL
STUDENT_COLUMN = "llama-3.1-8b-it"

# Judge LLM panel — uses canonical 3-judge panel
JUDGE_CONFIGS = dict(JUDGE_PANEL)

# Default single judge (used if runner doesn't support panel)
JUDGE_CONFIG = JudgeLLMConfig()
