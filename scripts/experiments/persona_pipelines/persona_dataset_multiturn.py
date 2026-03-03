#!/usr/bin/env python3
"""Generate a multi-turn persona dataset with assistant editing and responder turns."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from scripts.common.config import DatasetConfig, GenerationConfig
from scripts.common.persona_registry import (
    DEFAULT_PERSONA,
    PERSONA_DEFAULTS,
    get_persona_default_evaluations,
    get_persona_prompt_template,
)
from scripts.conversation_generation import (
    ConversationGenerationConfig,
    ResponderConfig,
    run_conversation_generation,
)
from scripts.datasets import export_dataset
from scripts.editing import EditingConfig, QualityConfig
from scripts.inference import InferenceConfig
from scripts.persona_metrics import PersonaMetricsConfig, run_persona_metrics


DEFAULT_DATASET = "vicgalle/alpaca-gpt4"
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_EDITOR_MODEL = "gpt-5-nano-2025-08-07"
DEFAULT_RESPONDER_MODEL = "gpt-5-nano-2025-08-07"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a multi-turn edited persona dataset.")
    parser.add_argument("--persona", type=str, default=DEFAULT_PERSONA, choices=sorted(PERSONA_DEFAULTS))
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--dataset-name", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--num-assistant-turns", type=int, default=3)
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--editor-model", type=str, default=DEFAULT_EDITOR_MODEL)
    parser.add_argument("--responder-model", type=str, default=DEFAULT_RESPONDER_MODEL)
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    load_dotenv()

    persona = args.persona
    prompt_template = get_persona_prompt_template(persona)
    evaluations = get_persona_default_evaluations(persona)
    run_id = args.run_id or f"{persona}-multiturn-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir = Path("scratch") / "runs" / run_id

    config = ConversationGenerationConfig(
        dataset=DatasetConfig(
            source="huggingface",
            name=args.dataset_name,
            split="train",
            max_samples=args.max_samples,
        ),
        run_dir=run_dir,
        num_assistant_turns=args.num_assistant_turns,
        assistant_inference=InferenceConfig(
            model=args.base_model,
            provider="local",
            dataset=DatasetConfig(
                source="huggingface",
                name=args.dataset_name,
                split="train",
                max_samples=args.max_samples,
            ),
            generation=GenerationConfig(max_new_tokens=256, batch_size=8, num_responses_per_prompt=1),
        ),
        editing=EditingConfig(
            provider="openai",
            model=args.editor_model,
            prompt_template=prompt_template,
            max_concurrent=8,
            quality=QualityConfig(
                enabled=False,
                evaluations=evaluations,
                persona=persona,
            ),
        ),
        responder=ResponderConfig(
            provider="openai",
            model=args.responder_model,
            prompt_template="natural_partner",
        ),
        editing_variant=f"{persona}_multiturn",
    )

    dataset, result = run_conversation_generation(config)
    print(f"Run dir: {run_dir}")
    print(f"Conversation export: {result.exports['conversation_training']}")
    print(f"Trace export: {result.exports['conversation_trace']}")
    print(f"Completed conversations: {result.num_completed}/{result.num_conversations}")
    print(
        f"Completed assistant turns: {result.num_assistant_turns_completed} / "
        f"{result.num_conversations * result.num_assistant_turns_target}"
    )

    if not args.skip_eval:
        eval_config = PersonaMetricsConfig(
            evaluations=evaluations,
            run_dir=run_dir,
            target_variant=config.editing_variant,
            output_path=run_dir / "exports" / "conversation_eval.jsonl",
        )
        _, eval_result = run_persona_metrics(eval_config)
        print(f"Eval output: {eval_result.output_path}")
        print(f"Eval aggregates: {eval_result.aggregates}")

    export_path = export_dataset(
        run_dir,
        profile="conversation_training",
        variant_name=config.editing_variant,
    )
    print(f"Final export: {export_path}")
    print(f"Dataset rows available inline: {len(dataset)}")


if __name__ == "__main__":
    main()
