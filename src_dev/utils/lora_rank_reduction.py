"""Reduce the rank of a saved LoRA adapter via truncated SVD.

Loads the adapter onto the base model, applies
:class:`src.utils.peft_manipulations.LoRaRankReducer` in-place, and saves the
rank-reduced adapter to a new directory. Idempotent on rerun.

CLI
---
    uv run python -m src_dev.utils.lora_rank_reduction \\
        --source-dir <path-to-full-rank-adapter> \\
        --target-dir <path-to-write> \\
        --new-rank 1 \\
        --base-model meta-llama/Llama-3.1-8B-Instruct
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from src.utils.peft_manipulations import LoRaRankReducer

logger = logging.getLogger(__name__)

# PEFT's save_pretrained writes directly to save_directory when the adapter
# is named "default"; any other name creates a <save_directory>/<name>/ subdir.
# Keeping it "default" avoids the flatten-after-save dance.
_ADAPTER_NAME = "default"


def _already_reduced(target_dir: Path, new_rank: int) -> bool:
    config_path = target_dir / "adapter_config.json"
    if not config_path.is_file():
        return False
    try:
        cfg = json.loads(config_path.read_text())
    except json.JSONDecodeError:
        return False
    return cfg.get("r") == new_rank


def reduce_adapter_rank_on_disk(
    source_dir: str | Path,
    target_dir: str | Path,
    new_rank: int,
    base_model: str,
    device_map: str = "auto",
    dtype: torch.dtype = torch.bfloat16,
) -> Path:
    """Load adapter onto *base_model*, SVD-reduce to *new_rank*, save to *target_dir*.

    Args:
        source_dir: Directory containing the full-rank LoRA adapter
            (``adapter_config.json`` + ``adapter_model.safetensors``).
        target_dir: Where the rank-reduced adapter should be written.
        new_rank: Target rank for truncated-SVD reduction (>= 1).
        base_model: HuggingFace model id (or local path) to load as the base
            for the temporary PeftModel.
        device_map: Passed to ``AutoModelForCausalLM.from_pretrained``.
        dtype: ``torch.dtype`` for the base model.

    Returns:
        The resolved ``target_dir`` path.

    Notes:
        Idempotent: if ``target_dir/adapter_config.json`` already has ``r ==
        new_rank``, this is a no-op.
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source adapter dir does not exist: {source_dir}")
    if not (source_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"Source adapter dir has no adapter_config.json: {source_dir}"
        )
    if new_rank < 1:
        raise ValueError(f"new_rank must be >= 1, got {new_rank}")

    if _already_reduced(target_dir, new_rank):
        logger.info(
            "Target dir %s already has r=%d; skipping reduction.", target_dir, new_rank
        )
        return target_dir

    logger.info("Loading base model %s (dtype=%s)", base_model, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map=device_map,
    )

    logger.info("Loading adapter from %s", source_dir)
    peft_model = PeftModel.from_pretrained(
        model, str(source_dir), adapter_name=_ADAPTER_NAME
    )
    peft_model.set_adapter(_ADAPTER_NAME)

    orig_rank = peft_model.peft_config[_ADAPTER_NAME].r
    if orig_rank < new_rank:
        raise ValueError(
            f"Source adapter rank {orig_rank} is smaller than requested "
            f"new_rank={new_rank}; nothing to reduce."
        )
    logger.info("Reducing rank from %d to %d via truncated SVD", orig_rank, new_rank)
    LoRaRankReducer(peft_model, _ADAPTER_NAME, new_rank=new_rank).apply()

    target_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(target_dir), selected_adapters=[_ADAPTER_NAME])
    logger.info("Saved rank-%d adapter to %s", new_rank, target_dir)

    # Release GPU memory before returning — the caller (an eval-config import)
    # is about to reload the base model for the sweep.
    del peft_model
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return target_dir


_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True,
                        help="Path to the saved full-rank LoRA adapter directory.")
    parser.add_argument("--target-dir", type=Path, required=True,
                        help="Destination for the rank-reduced adapter.")
    parser.add_argument("--new-rank", type=int, required=True,
                        help="Target rank for SVD reduction (>=1).")
    parser.add_argument("--base-model", type=str, required=True,
                        help="HuggingFace base-model id or local path.")
    parser.add_argument("--dtype", default="bfloat16", choices=list(_DTYPES))
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    reduce_adapter_rank_on_disk(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        new_rank=args.new_rank,
        base_model=args.base_model,
        device_map=args.device_map,
        dtype=_DTYPES[args.dtype],
    )


if __name__ == "__main__":
    _main()
