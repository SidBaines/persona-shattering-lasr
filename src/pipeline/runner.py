"""Pipeline orchestration for running extraction stages."""

from pathlib import Path


class PipelineRunner:
    """Orchestrates the persona extraction pipeline stages."""

    def __init__(self, config: dict):
        """Initialize the pipeline runner.

        Args:
            config: Pipeline configuration dictionary.
        """
        self.config = config
        self.stages = config.get("pipeline", {}).get("stages", [])
        self.output_dir = Path(config.get("pipeline", {}).get("output_dir", "scratch/run"))

    def run_all(self) -> None:
        """Run all pipeline stages in sequence."""
        raise NotImplementedError("Pipeline runner not yet implemented")

    def run_stage(self, stage_name: str) -> None:
        """Run a single pipeline stage.

        Args:
            stage_name: Name of the stage to run.
        """
        raise NotImplementedError(f"Stage '{stage_name}' not yet implemented")
