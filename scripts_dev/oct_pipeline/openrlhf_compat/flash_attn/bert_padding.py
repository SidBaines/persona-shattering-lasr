"""Small subset of flash_attn.bert_padding used by OpenRLHF.

The real flash-attn package is only required for sample packing / ring
attention. Our OCT runs do not enable packing, but OpenRLHF imports these
symbols eagerly, so we provide lightweight torch fallbacks.
"""

from __future__ import annotations

import torch


def rearrange(tensor: torch.Tensor, pattern: str) -> torch.Tensor:
    """Implement the single rearrange pattern OpenRLHF uses."""
    if pattern != "b s ... -> (b s) ...":
        raise NotImplementedError(f"Unsupported rearrange pattern in flash_attn shim: {pattern}")
    batch, seqlen = tensor.shape[:2]
    return tensor.reshape(batch * seqlen, *tensor.shape[2:])


def index_first_axis(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Index rows along the first axis."""
    return tensor.index_select(0, indices)


def unpad_input(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, None]:
    """Remove padded tokens, mirroring the flash-attn helper contract."""
    batch, seqlen = attention_mask.shape
    flat_states = hidden_states.reshape(batch * seqlen, *hidden_states.shape[2:])
    flat_mask = attention_mask.reshape(-1).to(dtype=torch.bool)
    indices = flat_mask.nonzero(as_tuple=False).squeeze(-1)
    unpadded = flat_states.index_select(0, indices)

    lengths = attention_mask.sum(dim=-1, dtype=torch.int32)
    cu_seqlens = torch.zeros(batch + 1, dtype=torch.int32, device=attention_mask.device)
    cu_seqlens[1:] = torch.cumsum(lengths, dim=0)
    max_seqlen = int(lengths.max().item()) if batch > 0 else 0
    return unpadded, indices, cu_seqlens, max_seqlen, None


def pad_input(
    hidden_states: torch.Tensor,
    indices: torch.Tensor,
    batch: int,
    seqlen: int,
) -> torch.Tensor:
    """Restore unpadded states back to (batch, seqlen, ...)."""
    output = hidden_states.new_zeros((batch * seqlen, *hidden_states.shape[1:]))
    output.index_copy_(0, indices, hidden_states)
    return output.reshape(batch, seqlen, *hidden_states.shape[1:])
