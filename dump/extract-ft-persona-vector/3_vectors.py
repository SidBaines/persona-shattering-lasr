#!/usr/bin/env python3
"""
Distill persona vectors from activations.

This script computes the mean activation vector across all responses,
representing the persona induced by the LoRA fine-tuning.

Unlike the assistant-axis pipeline which filters by judge scores,
we use all activations since the persona is already encoded in the LoRA weights.

Usage:
    python 3_vectors.py \
        --activations_file outputs/activations/gemma-test-20260211-221245.pt \
        --output_file outputs/vectors/gemma-test-20260211-221245.pt \
        --min_samples 50
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_activations(activations_file: Path) -> dict:
    """Load activations from .pt file."""
    return torch.load(activations_file, map_location="cpu", weights_only=False)


def compute_persona_vector(activations: dict, min_samples: int = 50) -> torch.Tensor:
    """
    Compute mean persona vector from all activations.

    Args:
        activations: Dict mapping keys to tensors (n_layers, hidden_dim)
        min_samples: Minimum number of samples required

    Returns:
        Mean vector of shape (n_layers, hidden_dim)
    """
    all_acts = list(activations.values())

    if len(all_acts) < min_samples:
        raise ValueError(f"Only {len(all_acts)} samples, need at least {min_samples}")

    logger.info(f"Computing persona vector from {len(all_acts)} samples")

    # Stack and compute mean
    stacked = torch.stack(all_acts)  # (n_samples, n_layers, hidden_dim)
    mean_vector = stacked.mean(dim=0)  # (n_layers, hidden_dim)

    return mean_vector


def main():
    parser = argparse.ArgumentParser(
        description='Distill persona vector from activations',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--activations_file', type=str, required=True,
                       help='Path to activations .pt file')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Output path for persona vector .pt file')
    parser.add_argument('--min_samples', type=int, default=50,
                       help='Minimum number of samples required (default: 50)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing output file')

    args = parser.parse_args()

    # Setup paths
    activations_file = Path(args.activations_file)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Check if output already exists
    if output_file.exists() and not args.overwrite:
        logger.info(f"Output file already exists: {output_file}")
        logger.info("Use --overwrite to regenerate")
        return

    # Load activations
    logger.info(f"Loading activations from {activations_file}")
    activations = load_activations(activations_file)
    logger.info(f"Loaded {len(activations)} activation samples")

    # Compute persona vector
    try:
        persona_vector = compute_persona_vector(activations, min_samples=args.min_samples)
        logger.info(f"Persona vector shape: {persona_vector.shape}")

        # Save with metadata
        checkpoint_name = activations_file.stem
        save_data = {
            "vector": persona_vector,
            "checkpoint": checkpoint_name,
            "n_samples": len(activations),
            "n_layers": persona_vector.shape[0],
            "hidden_dim": persona_vector.shape[1],
        }

        logger.info(f"Saving persona vector to {output_file}")
        torch.save(save_data, output_file)
        logger.info("Done!")

    except ValueError as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
