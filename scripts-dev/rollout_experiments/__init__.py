"""
Shared infrastructure for multi-phase rollout experiment scripts.

These are for evals across potentially long conversations (mainly to measure weight-based mitigation strategies e.g. applying LoRas)
"""

from scripts.experiments.rollout_experiments.utils import (
    Phase,
    RolloutExperimentConfig,
    UserSimulatorConfig,
    build_assistant_inference,
    build_dataset,
    build_user_simulator,
    evaluate_messages,
    export_evaluated_rollouts,
    export_rollouts,
    run_experiment,
    run_phased_rollout,
    save_experiment_metadata,
    upload_to_hf,
)

__all__ = [
    "Phase",
    "RolloutExperimentConfig",
    "UserSimulatorConfig",
    "build_assistant_inference",
    "build_dataset",
    "build_user_simulator",
    "evaluate_messages",
    "export_evaluated_rollouts",
    "export_rollouts",
    "run_experiment",
    "run_phased_rollout",
    "save_experiment_metadata",
    "upload_to_hf",
]
