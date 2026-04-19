"""Stage-orchestrator entry points for the psychometric FA pipeline.

Each ``run_stage_<name>(cfg, ...) -> <Stage>Result`` function takes its
config dataclass (see :mod:`src_dev.psychometric.config`) and runs the
corresponding pipeline stage. Stages are independent: a subset script can
call a single one against an existing run directory.
"""

from src_dev.psychometric.stages.factor_analysis import run_stage_factor_analysis
from src_dev.psychometric.stages.labeling import run_stage_labeling
from src_dev.psychometric.stages.questionnaire import run_stage_questionnaire
from src_dev.psychometric.stages.realism_judge import run_stage_realism_judge
from src_dev.psychometric.stages.rollouts import run_stage_rollouts
from src_dev.psychometric.stages.trait_scoring import run_stage_trait_scoring
from src_dev.psychometric.stages.validation import run_stage_validation

__all__ = [
    "run_stage_factor_analysis",
    "run_stage_labeling",
    "run_stage_questionnaire",
    "run_stage_realism_judge",
    "run_stage_rollouts",
    "run_stage_trait_scoring",
    "run_stage_validation",
]
