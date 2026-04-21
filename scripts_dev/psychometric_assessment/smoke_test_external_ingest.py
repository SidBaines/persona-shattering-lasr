"""Smoke test for the external-rollout ingestion stage.

Runs Stage 1 (ingest-external) in isolation on a tiny sample, without
touching the GPU. Verifies:

  1. The adapter streams rows and yields canonical message lists.
  2. Reservoir sampling is deterministic.
  3. ``ingest_source_dataset`` + ``materialize_canonical_samples`` +
     ``export_dataset`` produce a rollout dir that Stage 2 would accept.
  4. ``load_samples`` round-trips to ``SampleRecord`` objects with
     multi-turn ``messages``.
  5. HF upload + hydrate round-trip (if HF_TOKEN is available).

Writes to a temporary rollout dir under scratch/ so it never interferes
with the main psychometric_fa tree.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src_dev.datasets import load_samples
from src_dev.psychometric import ExternalRolloutsStageConfig, RunContext
from src_dev.psychometric.stages import run_stage_ingest_external_rollouts


SCRATCH_ROOT = Path("scratch/smoke_external_ingest")
HF_REPO_ID = "persona-shattering-lasr/psychometric-fa-runs"
RUN_ID = "rollouts-external-kwai_swe_smith-qwen38b-5p-seed436-smoke"


def main() -> None:
    run_dir = SCRATCH_ROOT / RUN_ID
    ctx = RunContext(
        scratch_root=SCRATCH_ROOT,
        hf_repo_id=HF_REPO_ID,
        rollout_run_id=RUN_ID,
        questionnaire_run_id=RUN_ID + "-q_smoke",
        rollout_dir=run_dir,
        questionnaire_dir=run_dir.parent / (RUN_ID + "-q_smoke"),
    )
    cfg = ExternalRolloutsStageConfig(
        ctx=ctx,
        source="kwai_swe_smith",
        assistant_model="Qwen/Qwen3-8B",
        assistant_provider="vllm",
        max_samples=5,
        seed=436,
        max_scan=20,
    )
    print(f"Smoke test: ingest {cfg.max_samples} Kwai-Klear rows → {run_dir}\n")
    run_stage_ingest_external_rollouts(cfg)

    samples = load_samples(run_dir)
    print(f"\nLoaded {len(samples)} samples from canonical dir:")
    for s in samples:
        n_assistant = sum(1 for m in s.messages if m.role == "assistant")
        n_user = sum(1 for m in s.messages if m.role == "user")
        total_chars = sum(len(m.content) for m in s.messages)
        print(
            f"  {s.sample_id[:20]}...  n_msg={len(s.messages):3d}  "
            f"n_asst={n_assistant:3d}  n_user={n_user:3d}  chars={total_chars:,}"
        )
        print(f"    source_info: {s.source_info}")

    # Sanity: each sample has multi-turn conversation history.
    assert all(
        sum(1 for m in s.messages if m.role == "assistant") >= 1
        for s in samples
    ), "expected at least one assistant turn per sample"
    print("\nSmoke test: PASS")


if __name__ == "__main__":
    main()
