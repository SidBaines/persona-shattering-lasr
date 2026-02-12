#!/usr/bin/env python3
"""
Generate rollouts from a model (optionally with LoRA fine-tuning).

This script loads a base model (optionally with a LoRA checkpoint) and generates
responses to a set of questions, saving them as JSONL files.

Usage (with LoRA):
    python 1_generate.py \
        --base_model google/gemma-2-2b-it \
        --lora_checkpoint scratch/gemma-test-20260211-221245/checkpoints/final \
        --questions_file ../../../assistant-axis/data/extraction_questions.jsonl \
        --output_dir outputs/rollouts \
        --question_count 240

Usage (base model only):
    python 1_generate.py \
        --base_model google/gemma-2-2b-it \
        --questions_file ../../../assistant-axis/data/extraction_questions.jsonl \
        --output_dir outputs/rollouts \
        --question_count 240
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict

import jsonlines
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scripts.common.config import GenerationConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_questions(questions_file: Path, max_count: int = None) -> List[Dict]:
    """Load questions from JSONL file."""
    questions = []
    with jsonlines.open(questions_file, 'r') as reader:
        for entry in reader:
            questions.append(entry)
            if max_count and len(questions) >= max_count:
                break
    return questions


def load_model(base_model: str, lora_checkpoint: str = None, dtype: str = "bfloat16", merge_lora: bool = False):
    """Load base model, optionally with LoRA checkpoint."""
    logger.info(f"Loading base model: {base_model}")

    # Convert dtype string to torch dtype
    torch_dtype = getattr(torch, dtype)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Optionally load LoRA checkpoint
    if lora_checkpoint:
        logger.info(f"Loading LoRA checkpoint: {lora_checkpoint}")
        model = PeftModel.from_pretrained(model, lora_checkpoint)

        # Optionally merge LoRA weights for 2x faster inference
        if merge_lora:
            logger.info("Merging LoRA weights for faster inference...")
            model = model.merge_and_unload()
        else:
            logger.info("Keeping LoRA as adapters (slower inference, less memory)")
    else:
        logger.info("Running with base model only (no LoRA)")

    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    question: str,
    generation_config: GenerationConfig,
) -> str:
    """Generate a single response to a question."""
    # Format as conversation
    messages = [{"role": "user", "content": question}]

    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=generation_config.max_new_tokens,
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            do_sample=generation_config.do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated part (excluding prompt)
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return response


def generate_batch_responses(
    model,
    tokenizer,
    questions: List[str],
    generation_config: GenerationConfig,
    batch_size: int = 8,
) -> List[str]:
    """Generate responses to multiple questions in batches."""
    all_responses = []

    for i in range(0, len(questions), batch_size):
        batch_questions = questions[i:i + batch_size]

        # Format all questions as conversations
        batch_prompts = []
        for question in batch_questions:
            messages = [{"role": "user", "content": question}]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            batch_prompts.append(prompt)

        # Tokenize batch with padding
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate batch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=generation_config.max_new_tokens,
                temperature=generation_config.temperature,
                top_p=generation_config.top_p,
                do_sample=generation_config.do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode each response (excluding prompts)
        for j, output in enumerate(outputs):
            prompt_len = inputs['input_ids'][j].shape[0]
            generated_ids = output[prompt_len:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            all_responses.append(response)

    return all_responses


def main():
    parser = argparse.ArgumentParser(
        description='Generate rollouts from LoRA fine-tuned model',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--base_model', type=str, required=True,
                       help='Base HuggingFace model name')
    parser.add_argument('--lora_checkpoint', type=str, default=None,
                       help='Path to LoRA checkpoint directory (optional, uses base model if not provided)')
    parser.add_argument('--questions_file', type=str, required=True,
                       help='Path to questions JSONL file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for rollouts')
    parser.add_argument('--question_count', type=int, default=240,
                       help='Number of questions to use (default: 240)')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                       help='Model dtype (default: bfloat16)')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                       help='Maximum tokens to generate (default: 512)')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Sampling temperature (default: 0.7)')
    parser.add_argument('--top_p', type=float, default=0.9,
                       help='Top-p sampling (default: 0.9)')
    parser.add_argument('--merge_lora', action='store_true',
                       help='Merge LoRA weights for 2x faster inference (uses more memory)')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for generation (default: 1, higher = faster but more memory)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing output file')

    args = parser.parse_args()

    # Setup paths
    questions_file = Path(args.questions_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine output filename from LoRA checkpoint name or base model
    if args.lora_checkpoint:
        checkpoint_name = Path(args.lora_checkpoint).parent.parent.name
    else:
        # Use base model name (last part after /)
        checkpoint_name = args.base_model.split('/')[-1] + "-base"
    output_file = output_dir / f"{checkpoint_name}.jsonl"

    # Check if output already exists
    if output_file.exists() and not args.overwrite:
        logger.info(f"Output file already exists: {output_file}")
        logger.info("Use --overwrite to regenerate")
        return

    # Load questions
    logger.info(f"Loading questions from {questions_file}")
    questions = load_questions(questions_file, max_count=args.question_count)
    logger.info(f"Loaded {len(questions)} questions")

    # Load model (with or without LoRA)
    model, tokenizer = load_model(
        args.base_model,
        lora_checkpoint=args.lora_checkpoint,
        dtype=args.dtype,
        merge_lora=args.merge_lora
    )

    # Setup generation config
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=True,
    )

    # Generate rollouts
    logger.info(f"Generating {len(questions)} rollouts with batch_size={args.batch_size}...")
    rollouts = []

    if args.batch_size > 1:
        # Batch generation mode
        batch_questions = [q["question"] for q in questions]

        for i in tqdm(range(0, len(questions), args.batch_size), desc="Generating batches"):
            batch = questions[i:i + args.batch_size]
            batch_q = [q["question"] for q in batch]

            try:
                responses = generate_batch_responses(
                    model, tokenizer, batch_q, generation_config, batch_size=args.batch_size
                )

                for question_data, response in zip(batch, responses):
                    question = question_data["question"]
                    question_id = question_data["id"]

                    conversation = [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": response}
                    ]

                    rollout = {
                        "conversation": conversation,
                        "question_index": question_id,
                        "question": question,
                        "response": response,
                    }
                    rollouts.append(rollout)

            except Exception as e:
                logger.error(f"Error generating batch starting at index {i}: {e}")
                continue
    else:
        # Single generation mode
        for question_data in tqdm(questions, desc="Generating"):
            question = question_data["question"]
            question_id = question_data["id"]

            try:
                response = generate_response(model, tokenizer, question, generation_config)

                # Format as conversation (matching assistant-axis format)
                conversation = [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": response}
                ]

                rollout = {
                    "conversation": conversation,
                    "question_index": question_id,
                    "question": question,
                    "response": response,
                }
                rollouts.append(rollout)

            except Exception as e:
                logger.error(f"Error generating response for question {question_id}: {e}")
                continue

    # Save rollouts
    logger.info(f"Saving {len(rollouts)} rollouts to {output_file}")
    with jsonlines.open(output_file, 'w') as writer:
        writer.write_all(rollouts)

    logger.info("Done!")


if __name__ == "__main__":
    main()
