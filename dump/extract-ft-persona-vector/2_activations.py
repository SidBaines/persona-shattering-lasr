#!/usr/bin/env python3
"""
Extract activations from rollouts.

This script loads rollouts (JSONL files) and extracts mean response activations
by re-running the model (optionally with LoRA) with activation hooks, saving them as .pt files.

NOTE: Uses assistant-axis compatible token extraction to exclude special tokens from activations.

Usage (with LoRA):
    python 2_activations.py \
        --base_model google/gemma-2-2b-it \
        --lora_checkpoint scratch/gemma-test-20260211-221245/checkpoints/final \
        --rollouts_file outputs/rollouts/gemma-test-20260211-221245.jsonl \
        --output_dir outputs/activations \
        --batch_size 16

Usage (base model only):
    python 2_activations.py \
        --base_model google/gemma-2-2b-it \
        --rollouts_file outputs/rollouts/gemma-2-2b-it-base.jsonl \
        --output_dir outputs/activations \
        --batch_size 16
"""

import argparse
import gc
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional

import jsonlines
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import response token utilities (assistant-axis compatible)
# Use relative import since __init__.py doesn't exist in this directory
from response_token_utils import get_assistant_response_token_ids

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_rollouts(rollouts_file: Path) -> List[Dict]:
    """Load rollouts from JSONL file."""
    rollouts = []
    with jsonlines.open(rollouts_file, 'r') as reader:
        for entry in reader:
            rollouts.append(entry)
    return rollouts


def load_model(base_model: str, lora_checkpoint: str = None, dtype: str = "bfloat16"):
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
        logger.info("Keeping LoRA as adapters (not merged) to save memory")
        # NOTE: We keep LoRA as adapters (not merged) to save memory.
        # This is slower than merging, but uses less GPU memory during loading.
        # To merge for faster inference (but more memory), uncomment:
        # logger.info("Merging LoRA weights...")
        # model = model.merge_and_unload()
    else:
        logger.info("Running with base model only (no LoRA)")

    return model, tokenizer


def extract_activations_batch(
    model,
    tokenizer,
    conversations: List[List[Dict[str, str]]],
    layers: Optional[List[int]] = None,
    batch_size: int = 16,
) -> List[Optional[torch.Tensor]]:
    """
    Extract mean response activations for a batch of conversations.

    Args:
        model: The language model
        tokenizer: The tokenizer
        conversations: List of conversations (each is a list of message dicts)
        layers: List of layer indices to extract (None = all layers)
        batch_size: Batch size for processing

    Returns:
        List of activation tensors, one per conversation
        Each tensor has shape (n_layers, hidden_dim)
    """
    if layers is None:
        # Extract from all layers
        n_layers = model.config.num_hidden_layers
        layers = list(range(n_layers))

    all_activations = []
    activations_storage = {}

    def create_hook(layer_idx):
        """Create a hook to capture activations at a specific layer."""
        def hook(module, input, output):
            # output is typically (batch_size, seq_len, hidden_dim)
            # or just the hidden states tensor
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # Store in dict by layer index
            if layer_idx not in activations_storage:
                activations_storage[layer_idx] = []
            activations_storage[layer_idx].append(hidden_states.detach().cpu())

        return hook

    # Register hooks on specified layers
    hooks = []
    for layer_idx in layers:
        # Access layer - this depends on model architecture
        # For PEFT models: model.base_model.model.layers[i]
        # For regular models: model.model.layers[i]

        # Try different layer access patterns based on model type
        if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
            # PEFT model: try base_model.model.model.layers first, then base_model.model.layers
            if hasattr(model.base_model.model, 'model') and hasattr(model.base_model.model.model, 'layers'):
                layer = model.base_model.model.model.layers[layer_idx]
            elif hasattr(model.base_model.model, 'layers'):
                layer = model.base_model.model.layers[layer_idx]
            else:
                raise ValueError(f"Cannot find layers in PEFT model: {type(model.base_model.model)}")
        elif hasattr(model, 'model'):
            # Regular model: try model.model.layers first, then model.layers
            if hasattr(model.model, 'layers'):
                layer = model.model.layers[layer_idx]
            elif hasattr(model, 'layers'):
                layer = model.layers[layer_idx]
            else:
                raise ValueError(f"Cannot find layers in model: {type(model)}")
        else:
            raise ValueError(f"Cannot access layers for model type: {type(model)}")

        hook = layer.register_forward_hook(create_hook(layer_idx))
        hooks.append(hook)

    # Process conversations in batches
    num_conversations = len(conversations)

    for batch_start in range(0, num_conversations, batch_size):
        batch_end = min(batch_start + batch_size, num_conversations)
        batch_conversations = conversations[batch_start:batch_end]

        # Clear activation storage for this batch
        activations_storage.clear()

        # Format conversations as prompts (including full conversation)
        prompts = []
        for conv in batch_conversations:
            prompt = tokenizer.apply_chat_template(
                conv,
                tokenize=False,
                add_generation_prompt=False
            )
            prompts.append(prompt)

        # Tokenize batch
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Forward pass (hooks will capture activations)
        with torch.no_grad():
            _ = model(**inputs)

        # Process activations for this batch
        # activations_storage[layer_idx] contains list of tensors for this batch
        # Each tensor is (batch_size, seq_len, hidden_dim)

        for batch_idx in range(len(batch_conversations)):
            conv = batch_conversations[batch_idx]

            # Use assistant-axis compatible token extraction
            # This excludes special tokens like <end_of_turn>, \n, <eos>
            try:
                _, response_start_idx, response_end_idx = get_assistant_response_token_ids(
                    tokenizer, conv
                )
            except Exception as e:
                logger.warning(f"Failed to extract response tokens for batch {batch_idx}: {e}")
                # Fallback to simple approach (will include special tokens)
                full_tokens = tokenizer(prompts[batch_idx], add_special_tokens=False)['input_ids']
                conv_without_last = conv[:-1] if len(conv) > 1 else []
                if conv_without_last:
                    prompt_without_assistant = tokenizer.apply_chat_template(
                        conv_without_last, tokenize=False, add_generation_prompt=True
                    )
                else:
                    prompt_without_assistant = tokenizer.apply_chat_template(
                        [], tokenize=False, add_generation_prompt=True
                    )
                prefix_tokens = tokenizer(prompt_without_assistant, add_special_tokens=False)['input_ids']
                response_start_idx = len(prefix_tokens)
                response_end_idx = len(full_tokens)

            # Get attention mask for this conversation
            attention_mask = inputs['attention_mask'][batch_idx]

            # Find the actual token positions (accounting for padding)
            # In left-padded sequences, tokens start after padding
            non_pad_start = (attention_mask == 1).nonzero(as_tuple=True)[0][0].item()

            # Map response indices to padded sequence positions
            response_start_padded = non_pad_start + response_start_idx
            response_end_padded = non_pad_start + response_end_idx

            # Ensure we don't go out of bounds
            seq_len = attention_mask.shape[0]
            response_start_padded = max(non_pad_start, min(response_start_padded, seq_len))
            response_end_padded = min(response_end_padded, seq_len)

            # Extract activations only from assistant response tokens
            layer_means = []
            for layer_idx in layers:
                layer_acts = activations_storage[layer_idx][0]  # (batch_size, seq_len, hidden_dim)
                conv_acts = layer_acts[batch_idx]  # (seq_len, hidden_dim)

                # Extract only assistant response positions
                response_acts = conv_acts[response_start_padded:response_end_padded]  # (response_len, hidden_dim)

                if response_acts.shape[0] > 0:
                    mean_act = response_acts.mean(dim=0)  # (hidden_dim,)
                else:
                    # Fallback: if we couldn't isolate response, use all valid tokens
                    logger.warning(f"Could not isolate assistant response for batch {batch_idx}, using all tokens")
                    valid_positions = attention_mask.bool()
                    valid_acts = conv_acts[valid_positions.cpu()]
                    mean_act = valid_acts.mean(dim=0)

                layer_means.append(mean_act)

            # Stack into (n_layers, hidden_dim)
            activations_tensor = torch.stack(layer_means)
            all_activations.append(activations_tensor)

        # Cleanup
        if batch_start % (batch_size * 5) == 0:
            torch.cuda.empty_cache()

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return all_activations


def main():
    parser = argparse.ArgumentParser(
        description='Extract activations from rollouts',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--base_model', type=str, required=True,
                       help='Base HuggingFace model name')
    parser.add_argument('--lora_checkpoint', type=str, default=None,
                       help='Path to LoRA checkpoint directory (optional, uses base model if not provided)')
    parser.add_argument('--rollouts_file', type=str, required=True,
                       help='Path to rollouts JSONL file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for activations')
    parser.add_argument('--layers', type=str, default='all',
                       help='Layers to extract (all or comma-separated indices)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for processing (default: 16)')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                       help='Model dtype (default: bfloat16)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing output file')

    args = parser.parse_args()

    # Setup paths
    rollouts_file = Path(args.rollouts_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine output filename from rollouts file
    checkpoint_name = rollouts_file.stem
    output_file = output_dir / f"{checkpoint_name}.pt"

    # Check if output already exists
    if output_file.exists() and not args.overwrite:
        logger.info(f"Output file already exists: {output_file}")
        logger.info("Use --overwrite to regenerate")
        return

    # Load rollouts
    logger.info(f"Loading rollouts from {rollouts_file}")
    rollouts = load_rollouts(rollouts_file)
    logger.info(f"Loaded {len(rollouts)} rollouts")

    # Load model (with or without LoRA)
    model, tokenizer = load_model(
        args.base_model,
        lora_checkpoint=args.lora_checkpoint,
        dtype=args.dtype
    )

    # Determine layers to extract
    n_layers = model.config.num_hidden_layers
    if args.layers == 'all':
        layers = list(range(n_layers))
    else:
        layers = [int(x.strip()) for x in args.layers.split(',')]

    logger.info(f"Model has {n_layers} layers")
    logger.info(f"Extracting from {len(layers)} layers")

    # Extract conversations
    conversations = [r["conversation"] for r in rollouts]

    # Extract activations
    logger.info(f"Extracting activations from {len(conversations)} conversations...")
    activations_list = extract_activations_batch(
        model=model,
        tokenizer=tokenizer,
        conversations=conversations,
        layers=layers,
        batch_size=args.batch_size,
    )

    # Build activations dict (keyed by question_index)
    activations_dict = {}
    for rollout, acts in zip(rollouts, activations_list):
        if acts is not None:
            key = f"q{rollout['question_index']}"
            activations_dict[key] = acts

    # Save activations
    logger.info(f"Saving {len(activations_dict)} activations to {output_file}")
    torch.save(activations_dict, output_file)

    # Cleanup
    gc.collect()
    torch.cuda.empty_cache()

    logger.info("Done!")


if __name__ == "__main__":
    main()
