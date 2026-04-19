"""Stage-orchestrator entry points for the psychometric FA pipeline.

Each ``run_stage_<name>(cfg) -> <Stage>Result`` function takes a single
config dataclass (see ``src_dev.psychometric.config``) and runs the
corresponding pipeline stage. Stages are independent: a subset script can
call a single one against an existing run directory.
"""
