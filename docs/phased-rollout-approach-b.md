# Phased Rollout via Context Handoff (Approach B)

## Problem

We want multi-phase rollout experiments where different phases use different models or LoRA
adapters — for example, "create context with o-enjoying LoRA, then observe the base model's
response given that context". Loading multiple models simultaneously is infeasible on a single
GPU, and coordinating model swaps across concurrent rollout jobs requires non-trivial resource
management (reference counting, locks, etc.).

## Solution: Separate runs with context handoff

Each phase is a fully independent rollout job with its own `run_dir`. Phase 2 seeds from the
completed conversations of Phase 1, so the prior conversation is in context. No coordination
between jobs is needed — Phase 1 completes fully before Phase 2 begins, each loads exactly one
model, and Phase 1 data can be reused as the seed for many different Phase 2 experiments.

```
Phase 1 run_dir/            Phase 2 run_dir/
  rollouts.jsonl    ──────►   sample_inputs.jsonl  (seeded from phase 1 conversations)
  (o-enjoying LoRA)             (base model, eval turns)
```

Negative turn indexing on `MessageSelector` means evaluation already targets only Phase 2 turns
naturally.

## What needs building

### 1. `ingest_rollout_as_seed` (new function in `scripts/datasets/`)

A function that reads a completed rollout's conversations and writes them as
`sample_inputs.jsonl` for a new run_dir. Each sample in Phase 2 gets the full Phase 1
conversation as its seed messages, with `source_stage` set to `"seed"` so the evaluation layer
treats them as context rather than observations.

**Signature (proposed):**

```python
def ingest_rollout_as_seed(
    source_run_dir: Path,
    dest_run_dir: Path,
    *,
    rollout_index: int | None = None,   # None = all rollouts; int = specific rollout only
    source_info: dict | None = None,
    responses_per_input: int = 1,
    overwrite: bool = False,
) -> None:
    ...
```

**What it does:**

1. Calls `materialize_canonical_samples(source_run_dir)` and `load_samples(source_run_dir)`
2. Groups samples by `input_group_id` to recover the per-seed rollout set
3. For each rollout (or the selected `rollout_index`), builds a list of `CanonicalMessage`
   objects from all messages in that conversation (excluding the seed user message if desired,
   or keeping it — configurable)
4. Re-tags all message metadata with `"source_stage": "seed"` so downstream evaluation
   excludes them from scoring
5. Writes one `SampleRecord` per conversation into `dest_run_dir/sample_inputs.jsonl` via
   the existing `init_run` + canonical write path

**Key detail:** `ingest_source_dataset` already supports a `"messages"` field in each dataset
row — it passes the list of message dicts directly into `CanonicalInput`. `ingest_rollout_as_seed`
can use this path, serialising the Phase 1 conversation into the `messages` field format the
existing ingestion code already handles. No changes to the ingestion schema are needed.

### 2. `RolloutGenerationConfig.seed_run_dir` (optional new field)

Alternatively (or additionally), `RolloutGenerationConfig` could accept a `seed_run_dir: Path | None`
field. When set, `run_rollout_generation` calls `ingest_rollout_as_seed` instead of
`ingest_source_dataset`. This keeps the experiment script interface clean:

```python
# Phase 1
phase1_config = RolloutGenerationConfig(
    dataset=DatasetConfig(...),
    run_dir=Path("scratch/phase1"),
    num_assistant_turns=5,
    assistant_inference=lora_inference_config,
    system_prompt=O_ENJOYING_PROMPT,
)
_, _ = run_rollout_generation(phase1_config)

# Phase 2 — seeds from Phase 1, different model
phase2_config = RolloutGenerationConfig(
    seed_run_dir=Path("scratch/phase1"),   # ← new field
    run_dir=Path("scratch/phase2"),
    num_assistant_turns=2,
    assistant_inference=base_model_inference_config,
)
_, _ = run_rollout_generation(phase2_config)
```

This is the minimal-change path. Alternatively, keep `ingest_rollout_as_seed` as a standalone
utility and call it explicitly in experiment scripts (more explicit, fits the project's
"no pipeline orchestrator" principle better).

### 3. Experiment orchestrator extension (`rollout_experiments/__init__.py`)

`Phase` and `run_phased_rollout` can be extended to wire up Approach B transparently:

```python
@dataclass
class Phase:
    num_turns: int
    assistant_system_prompt: str | None = None
    user_simulator: UserSimulatorConfig | None = None
    assistant_inference: InferenceConfig | None = None   # per-phase model override
    seed_from_previous: bool = False                     # use prior phase's run_dir as seed
```

When `seed_from_previous=True`, `run_phased_rollout` creates a fresh `run_dir` for this phase
and calls `ingest_rollout_as_seed(previous_run_dir, this_run_dir)` before running generation.

## What does NOT need to change

- `materialize_canonical_samples` — event-driven, model provenance is already per-message
- `MessageSelector` / `run_conversation_metrics` — negative turn indexing already scopes eval
  to Phase 2 turns naturally
- The evaluation pipeline — it reads from `run_dir` agnostically; pointing it at the Phase 2
  `run_dir` gives the right scope
- HuggingFace upload — each phase's `run_dir` can be uploaded independently

## Open questions to discuss with the team

1. **Rollout selection for seeding:** Should Phase 2 seed from all rollouts of Phase 1 (one
   Phase-2 conversation per Phase-1 rollout), or from a single representative rollout? The
   `rollout_index` param above supports both.

2. **Seed boundary in evaluation:** When scoring Phase 2 turns, should the Phase 1 messages
   be entirely invisible to evaluators (stripped from context passed to the judge), or should
   the judge receive them as context? The `preceding_content` field in `eval_items` currently
   carries only the immediately preceding message. If the judge needs the full Phase 1 history
   this would need extending.

3. **Conversation lineage display:** The Phase 2 `run_dir` will contain conversations that
   "start" mid-way through a longer exchange. The `rollouts.jsonl` export currently shows
   messages from turn 0 — it should probably include provenance metadata pointing at the
   Phase 1 `run_dir`.

4. **Multi-rollout fan-out:** Phase 1 might produce N rollouts per seed. Phase 2 could run
   K new rollouts on top of each Phase-1 rollout — is `N×K` fan-out ever useful, or should
   Phase 2 always be 1 rollout per Phase-1 conversation?

## Implementation order

1. `ingest_rollout_as_seed` standalone utility (self-contained, testable)
2. Tests for the new function
3. `Phase.seed_from_previous` + `Phase.assistant_inference` in the experiment orchestrator
4. A concrete experiment script (e.g. `o_frequency_phased.py`) that demonstrates the pattern
