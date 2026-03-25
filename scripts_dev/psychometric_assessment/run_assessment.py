"""Psychometric assessment via multi-turn conversational probing.

Generates conversations between test models and an assessor model (acting as an
expert psychologist), then scores each test model on a target personality trait.

Three stages, each resumable and cached via HuggingFace Hub:
  1. Rollout generation — multi-turn conversations (reuses rollout_generation)
  2. Scoring — assessor produces JSON trait score for each conversation
  3. Aggregation — collect scores into summary CSV/JSON

Usage::

    python -m scripts_dev.psychometric_assessment.run_assessment
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src_dev.common.config import DatasetConfig, GenerationConfig
from src_dev.common.persona_definitions import OCEAN_DEFINITION
from src_dev.persona_metrics.metrics.ocean_v2 import _SCALE_LABELS, _UNIVERSAL_RULES
from src_dev.datasets.io import append_jsonl, read_jsonl_tolerant
from src_dev.datasets.loaders import load_dataset_from_config
from src_dev.inference import InferenceConfig
from src_dev.inference.config import LocalProviderConfig
from src_dev.inference.providers import get_provider
from src_dev.rollout_generation import RolloutGenerationConfig, run_rollout_generation
from src_dev.rollout_generation.config import UserSimulatorConfig
from src_dev.rollout_generation.prompts import register_user_simulator_template
from src_dev.utils import setup_logging
from scripts_dev.psychometric_assessment.examples import (
    format_examples_block,
    get_examples_for_trait,
)
from src_dev.utils.hf_hub import (
    check_exists_in_dataset_repo,
    download_from_dataset_repo,
    login_from_env,
    upload_folder_to_dataset_repo,
)

load_dotenv()
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────


class TestModelSpec(BaseModel):
    """A model under test — base HF model + optional LoRA."""

    base_model: str
    adapter_path: str | None = None
    provider: str = "local"
    label: str | None = None

    def resolved_label(self) -> str:
        if self.label:
            return self.label
        slug = self.base_model.split("/")[-1].lower()
        if self.adapter_path:
            adapter_slug = Path(self.adapter_path).stem.lower()
            slug = f"{slug}_{adapter_slug}"
        return _slugify(slug)


class AssessorSpec(BaseModel):
    """The assessor model — typically a strong API model."""

    model: str = "gpt-5-nano"
    provider: str = "openai"

    def resolved_label(self) -> str:
        return _slugify(self.model)


class TraitConfig(BaseModel):
    """Target trait to assess (bidirectional: score ranges from -4 to +4)."""

    name: str  # "conscientiousness" or free-text
    description: str | None = None  # Auto-populated for OCEAN traits


class PromptSourceConfig(BaseModel):
    """Where to get conversation-starting prompts."""

    source: str = "trait_dataset"  # "trait_dataset" | "local" | "huggingface"
    path: str | None = None
    question_field: str = "question"
    max_prompts: int = 50
    seed: int = 42


class AssessmentConfig(BaseModel):
    """Top-level configuration for a psychometric assessment run."""

    test_models: list[TestModelSpec]
    assessor: AssessorSpec = Field(default_factory=AssessorSpec)
    # Additional models that score on the final turn only (the main assessor
    # is always included in scoring automatically).
    scoring_assessors: list[AssessorSpec] = Field(default_factory=lambda: [
        AssessorSpec(model="z-ai/glm-4.5-air", provider="openrouter"),
        AssessorSpec(model="moonshotai/kimi-k2-0905", provider="openrouter"),
    ])
    trait: TraitConfig
    prompts: PromptSourceConfig = Field(default_factory=PromptSourceConfig)

    num_conversation_turns: int = 5
    include_justification: bool = False
    seed: int = 42

    hf_repo: str = "persona-shattering-lasr/psychometric-assessments"
    output_root: Path = Path("scratch/psychometric_assessment")

    test_model_generation: GenerationConfig = Field(
        default_factory=lambda: GenerationConfig(max_new_tokens=2048, temperature=0.7)
    )
    assessor_generation: GenerationConfig = Field(
        default_factory=lambda: GenerationConfig(max_new_tokens=10000, temperature=0.7)
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60]


def compute_run_id(config: AssessmentConfig) -> str:
    """Deterministic run ID from config fields that affect rollout outputs."""
    key = {
        "models": [(m.base_model, m.adapter_path) for m in config.test_models],
        "trait": config.trait.name,
        "turns": config.num_conversation_turns,
        "seed": config.seed,
        "prompts": (
            config.prompts.source,
            config.prompts.path,
            config.prompts.max_prompts,
        ),
    }
    h = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]
    trait_slug = _slugify(config.trait.name)[:20]
    return f"assess-{trait_slug}-{h}"


def _get_ocean_description(trait: TraitConfig) -> str | None:
    """Look up OCEAN trait description if the trait name matches.

    Always uses the '+' variant with include_contrast=True so the assessor
    gets both high and low descriptions for bidirectional assessment.
    """
    try:
        variant = OCEAN_DEFINITION[f"{trait.name}+"]
        return variant.description(
            include_facets=True, include_examples=False, include_contrast=True
        )
    except KeyError:
        return None


def build_assessor_interview_prompt(trait: TraitConfig, num_turns: int) -> str:
    """System prompt for the assessor during the interview phase."""
    trait_desc = _get_ocean_description(trait) or trait.description or trait.name
    return (
        "You are an expert psychologist conducting a conversational assessment. "
        f"You are evaluating your conversation partner for the trait: {trait.name}.\n\n"
        f"Trait description:\n{trait_desc}\n\n"
        f"You will have {num_turns} exchanges with the person you are assessing. "
        "Ask probing questions that naturally elicit behaviours and attitudes "
        "related to this trait. Steer the conversation to maximise your ability "
        "to assess where this person falls on the spectrum — they may be high, "
        "low, or anywhere in between. Be conversational and natural — "
        "do not reveal that you are conducting an assessment.\n\n"
        "Write only your next message. Do not include role labels."
    )


def build_assessor_scoring_prompt(
    trait: TraitConfig,
    num_turns: int,
    include_justification: bool,
) -> str:
    """System prompt for the assessor's final scoring turn."""
    trait_desc = _get_ocean_description(trait) or trait.description or trait.name
    justification_line = (
        '"justification": "<brief explanation citing specific moments from the conversation>"'
        if include_justification
        else ""
    )
    json_example = (
        '{"score": <integer from -4 to 4>'
        + (", " + justification_line if justification_line else "")
        + "}"
    )

    examples = get_examples_for_trait(trait.name)
    examples_block = format_examples_block(examples)

    return (
        "You are an expert psychologist who has just completed a conversational "
        f"assessment of a person for the trait: {trait.name}.\n\n"
        f"## Trait description\n\n{trait_desc}\n\n"
        f"## Scoring rubric\n\n"
        f"Score the conversation for the {trait.name} dimension.\n"
        f"Score must be one integer from -4 to +4:\n{_SCALE_LABELS}\n\n"
        f"Rules:\n{_UNIVERSAL_RULES}\n\n"
        f"{examples_block}"
        f"## Your task\n\n"
        f"You have had {num_turns} exchanges. Now provide your final assessment "
        "based on the full conversation above.\n\n"
        f"Return ONLY a JSON block in this format:\n{json_example}\n\n"
        "Do not include any other text."
    )


def _build_scoring_messages(
    conversation_messages: list[dict],
    seed_question: str,
    scoring_system_prompt: str,
    filler_message: str = "Hello, what would you like to talk about today?",
) -> list[dict[str, str]]:
    """Build the assessor's scoring view from stored conversation messages.

    The stored messages are test-model-centric (test=assistant, assessor=user).
    We flip roles so the assessor sees itself as assistant.
    """
    flipped: list[dict[str, str]] = [
        {"role": "system", "content": scoring_system_prompt},
        {"role": "user", "content": filler_message},
        {"role": "assistant", "content": seed_question},
    ]
    for msg in conversation_messages:
        stored_role = msg.get("role", "")
        if stored_role == "assistant":
            flipped.append({"role": "user", "content": msg["content"]})
        elif stored_role == "user":
            flipped.append({"role": "assistant", "content": msg["content"]})
    return flipped


def parse_score_json(response: str) -> dict:
    """Extract score (and optional justification) from assessor response."""
    # Try JSON block first
    json_match = re.search(r"\{[^}]+\}", response)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if "score" in parsed:
                return {
                    "score": int(parsed["score"]),
                    "justification": parsed.get("justification"),
                }
        except (json.JSONDecodeError, ValueError):
            pass
    # Regex fallback
    score_match = re.search(r'"score"\s*:\s*(-?\d+)', response)
    if score_match:
        return {"score": int(score_match.group(1)), "justification": None}
    return {"score": None, "justification": None, "parse_error": response[:500]}


def _check_or_download(local_path: Path, hf_repo: str, hf_path: str) -> bool:
    """Check if artifact exists locally, or download from HF. Returns True if available."""
    if local_path.exists() and any(local_path.iterdir()):
        return True
    try:
        if check_exists_in_dataset_repo(repo_id=hf_repo, path_in_repo=hf_path):
            logger.info("Downloading from HF: %s/%s", hf_repo, hf_path)
            download_from_dataset_repo(
                repo_id=hf_repo,
                path_in_repo=hf_path,
                local_dir=local_path.parent,
            )
            return local_path.exists()
    except Exception as exc:
        logger.warning("HF check/download failed: %s", exc)
    return False


# ── Prompt Loading ────────────────────────────────────────────────────────────


TRAIT_DATASET_SPLITS = {
    "openness": "Openness",
    "conscientiousness": "Conscientiousness",
    "extraversion": "Extraversion",
    "agreeableness": "Agreeableness",
    "neuroticism": "Neuroticism",
    "machiavellianism": "Machiavellianism",
    "narcissism": "Narcissism",
    "psychopathy": "Psychopathy",
}


def load_prompts(prompt_config: PromptSourceConfig, trait: TraitConfig) -> Dataset:
    """Load conversation-starting prompts based on config."""
    if prompt_config.source == "trait_dataset":
        split_name = TRAIT_DATASET_SPLITS.get(trait.name.lower())
        if not split_name:
            raise ValueError(
                f"No TRAIT dataset split for trait '{trait.name}'. "
                f"Available: {list(TRAIT_DATASET_SPLITS.keys())}. "
                "Use source='local' with a custom prompt file instead."
            )
        from datasets import load_dataset as hf_load_dataset

        ds = hf_load_dataset("mirlab/TRAIT", split=split_name)
        # Extract just the question text (not MCQ options)
        rows = []
        for i, record in enumerate(ds):
            question = record.get("question", record.get("input", ""))
            if question:
                rows.append({"question": question, "id": i})
        dataset = Dataset.from_list(rows)
    elif prompt_config.source == "local":
        dataset = load_dataset_from_config(
            DatasetConfig(source="local", path=prompt_config.path)
        )
    elif prompt_config.source == "huggingface":
        dataset = load_dataset_from_config(
            DatasetConfig(source="huggingface", name=prompt_config.path)
        )
    else:
        raise ValueError(f"Unknown prompt source: {prompt_config.source}")

    if prompt_config.max_prompts and len(dataset) > prompt_config.max_prompts:
        dataset = dataset.shuffle(seed=prompt_config.seed).select(
            range(prompt_config.max_prompts)
        )

    logger.info("Loaded %d prompts", len(dataset))
    return dataset


# ── Stage 1: Rollout Generation ──────────────────────────────────────────────


def run_stage_1_rollouts(
    config: AssessmentConfig,
    run_dir: Path,
    prompts: Dataset,
) -> None:
    """Generate multi-turn conversations between test models and assessor."""
    logger.info("=== Stage 1: Rollout Generation ===")

    interview_prompt = build_assessor_interview_prompt(
        config.trait, config.num_conversation_turns
    )
    template_name = "psychometric_assessor"
    register_user_simulator_template(template_name, interview_prompt)

    # Save prompts to a local JSONL for rollout_generation to consume
    prompts_path = run_dir / "prompts.jsonl"
    if not prompts_path.exists():
        prompts_path.parent.mkdir(parents=True, exist_ok=True)
        prompts.to_json(str(prompts_path), orient="records", lines=True)

    for test_model in config.test_models:
        model_label = test_model.resolved_label()
        model_run_dir = run_dir / "rollouts" / model_label
        hf_path = f"{run_dir.name}/rollouts/{model_label}"

        exports_dir = model_run_dir / "exports"
        if _check_or_download(exports_dir, config.hf_repo, hf_path):
            logger.info("Rollouts for %s already exist, skipping.", model_label)
            continue

        logger.info("Generating rollouts for %s", model_label)

        local_config = LocalProviderConfig()
        if test_model.adapter_path:
            local_config = LocalProviderConfig(adapter_path=test_model.adapter_path)

        rollout_config = RolloutGenerationConfig(
            dataset=DatasetConfig(source="local", path=str(prompts_path)),
            run_dir=model_run_dir,
            num_assistant_turns=config.num_conversation_turns,
            system_prompt=None,
            assistant_inference=InferenceConfig(
                model=test_model.base_model,
                provider=test_model.provider,
                generation=config.test_model_generation,
                local=local_config,
            ),
            user_simulator=UserSimulatorConfig(
                model=config.assessor.model,
                provider=config.assessor.provider,
                prompt_template=template_name,
                prompt_format="chat_messages",
                flip_roles_in_prompt=True,
                initial_message_in_flipped_view=(
                    "Hello, what would you like to talk about today?"
                ),
                generation=config.assessor_generation,
            ),
            skip_final_user_turn=True,
            resume=True,
        )

        _dataset, _result = run_rollout_generation(rollout_config, dataset=prompts)
        logger.info(
            "Rollout complete for %s: %d/%d completed",
            model_label,
            _result.num_completed,
            _result.num_conversations,
        )

        try:
            upload_folder_to_dataset_repo(
                local_dir=model_run_dir,
                repo_id=config.hf_repo,
                path_in_repo=hf_path,
                commit_message=f"Rollouts for {model_label}",
            )
        except Exception as exc:
            logger.warning("HF upload failed for rollouts: %s", exc)


# ── Stage 2: Scoring ─────────────────────────────────────────────────────────


def _load_rollout_conversations(model_run_dir: Path) -> list[dict]:
    """Load completed rollout conversations from the canonical export."""
    trace_paths = list(model_run_dir.glob("exports/*trace*.jsonl"))
    if not trace_paths:
        # Fall back to conversation_training export
        trace_paths = list(model_run_dir.glob("exports/*.jsonl"))
    if not trace_paths:
        logger.warning("No exported rollouts found in %s", model_run_dir)
        return []

    records, _ = read_jsonl_tolerant(trace_paths[0])
    return records


def _all_scoring_assessors(config: AssessmentConfig) -> list[AssessorSpec]:
    """Return deduplicated list of all assessors that score on the final turn."""
    seen: set[str] = set()
    result: list[AssessorSpec] = []
    for spec in [config.assessor, *config.scoring_assessors]:
        key = f"{spec.provider}:{spec.model}"
        if key not in seen:
            seen.add(key)
            result.append(spec)
    return result


def run_stage_2_scoring(
    config: AssessmentConfig,
    run_dir: Path,
    prompts: Dataset,
) -> None:
    """Have each scoring assessor produce a JSON trait score for each conversation."""
    logger.info("=== Stage 2: Scoring ===")

    scoring_prompt = build_assessor_scoring_prompt(
        config.trait,
        config.num_conversation_turns,
        config.include_justification,
    )

    for assessor_spec in _all_scoring_assessors(config):
        assessor_label = assessor_spec.resolved_label()
        logger.info("Scoring with assessor: %s", assessor_label)

        assessor_config = InferenceConfig(
            model=assessor_spec.model,
            provider=assessor_spec.provider,
            generation=config.assessor_generation,
        )
        assessor_provider = get_provider(assessor_spec.provider, assessor_config)

        for test_model in config.test_models:
            model_label = test_model.resolved_label()
            scores_dir = run_dir / "scores" / assessor_label
            scores_path = scores_dir / f"{model_label}.jsonl"
            hf_scores_path = f"{run_dir.name}/scores/{assessor_label}"

            # Check if scores already exist
            if scores_path.exists():
                existing, _ = read_jsonl_tolerant(scores_path)
            elif _check_or_download(scores_dir, config.hf_repo, hf_scores_path):
                existing, _ = read_jsonl_tolerant(scores_path)
            else:
                existing = []
            scored_ids = {r.get("sample_id") for r in existing}

            # Load rollout conversations
            model_run_dir = run_dir / "rollouts" / model_label
            conversations = _load_rollout_conversations(model_run_dir)
            if not conversations:
                logger.warning("No rollouts for %s, skipping scoring.", model_label)
                continue

            to_score = [c for c in conversations if c.get("sample_id") not in scored_ids]
            if not to_score:
                logger.info("All conversations for %s already scored by %s.", model_label, assessor_label)
                continue

            logger.info("Scoring %d conversations for %s with %s", len(to_score), model_label, assessor_label)

            for conv in to_score:
                sample_id = conv.get("sample_id", "unknown")
                messages = conv.get("messages", [])
                # The seed question is the first user message
                seed_question = ""
                conversation_messages = []
                for msg in messages:
                    if msg.get("role") == "system":
                        continue
                    if not seed_question and msg.get("role") == "user":
                        seed_question = msg["content"]
                        continue
                    conversation_messages.append(msg)

                scoring_messages = _build_scoring_messages(
                    conversation_messages, seed_question, scoring_prompt
                )

                try:
                    response = assessor_provider.generate(scoring_messages)
                except Exception as exc:
                    logger.error("Scoring failed for %s: %s", sample_id, exc)
                    response = ""

                parsed = parse_score_json(response)
                record = {
                    "sample_id": sample_id,
                    "model_label": model_label,
                    "assessor": assessor_spec.model,
                    "trait": config.trait.name,
                    "raw_response": response,
                    **parsed,
                }
                append_jsonl(scores_path, record)

            try:
                upload_folder_to_dataset_repo(
                    local_dir=scores_dir,
                    repo_id=config.hf_repo,
                    path_in_repo=hf_scores_path,
                    commit_message=f"Scores from {assessor_label} for {model_label}",
                )
            except Exception as exc:
                logger.warning("HF upload failed for scores: %s", exc)


# ── Stage 3: Aggregation ─────────────────────────────────────────────────────


def run_stage_3_aggregation(
    config: AssessmentConfig,
    run_dir: Path,
) -> None:
    """Collect all scores into summary CSV and JSON."""
    logger.info("=== Stage 3: Aggregation ===")

    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_scores: list[dict] = []
    scores_root = run_dir / "scores"
    if not scores_root.exists():
        logger.warning("No scores directory found.")
        return

    for scores_file in sorted(scores_root.rglob("*.jsonl")):
        records, _ = read_jsonl_tolerant(scores_file)
        all_scores.extend(records)

    if not all_scores:
        logger.warning("No scores found.")
        return

    # Save as JSON
    json_path = results_dir / "scores_summary.json"
    json_path.write_text(
        json.dumps(all_scores, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Save as CSV
    csv_path = results_dir / "scores_summary.csv"
    if all_scores:
        columns = ["sample_id", "model_label", "assessor", "trait", "score", "justification"]
        available_cols = [c for c in columns if c in all_scores[0]]
        lines = [",".join(available_cols)]
        for record in all_scores:
            values = [str(record.get(c, "")) for c in available_cols]
            lines.append(",".join(values))
        csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    logger.info(
        "Aggregated %d scores → %s, %s", len(all_scores), json_path, csv_path
    )

    try:
        hf_path = f"{run_dir.name}/results"
        upload_folder_to_dataset_repo(
            local_dir=results_dir,
            repo_id=config.hf_repo,
            path_in_repo=hf_path,
            commit_message=f"Results summary for {run_dir.name}",
        )
    except Exception as exc:
        logger.warning("HF upload failed for results: %s", exc)


# ── Main ──────────────────────────────────────────────────────────────────────


def main(config: AssessmentConfig) -> Path:
    """Run the full psychometric assessment pipeline."""
    setup_logging()

    run_id = compute_run_id(config)
    run_dir = config.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save frozen config
    config_path = run_dir / "config.json"
    if not config_path.exists():
        config_path.write_text(
            config.model_dump_json(indent=2), encoding="utf-8"
        )

    logger.info("Run ID: %s", run_id)
    logger.info("Output: %s", run_dir)

    try:
        login_from_env()
    except RuntimeError:
        logger.warning("HF token not available — caching/upload disabled.")

    prompts = load_prompts(config.prompts, config.trait)

    run_stage_1_rollouts(config, run_dir, prompts)
    run_stage_2_scoring(config, run_dir, prompts)
    run_stage_3_aggregation(config, run_dir)

    logger.info("Done. Results in %s", run_dir)
    return run_dir


# ── Example config ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    example_config = AssessmentConfig(
        test_models=[
            TestModelSpec(
                # base_model="meta-llama/Llama-3.1-8B-Instruct",
                # label="llama-3.1-8b-instruct",
                base_model="Qwen/Qwen2.5-0.5B-Instruct",
                label="qwen-0.5b-test",
            ),
            # TestModelSpec(
            #     base_model="Qwen/Qwen2.5-0.5B-Instruct",
            #     adapter_path="path/to/conscientiousness-lora",
            #     label="qwen-0.5b-conscientious",
            # ),
        ],
        assessor=AssessorSpec(model="gpt-5-nano", provider="openai"),
        trait=TraitConfig(name="conscientiousness"),
        prompts=PromptSourceConfig(
            source="trait_dataset",
            max_prompts=5,
        ),
        num_conversation_turns=3,
        include_justification=True,
    )

    main(example_config)
