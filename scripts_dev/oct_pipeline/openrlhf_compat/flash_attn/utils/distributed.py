"""Subset of flash_attn.utils.distributed used by OpenRLHF."""

from __future__ import annotations

import torch
import torch.distributed as dist


def all_gather(tensor: torch.Tensor, group=None) -> torch.Tensor:
    """Gather tensors across ranks, or return the input unchanged for rank-1."""
    if not dist.is_available() or not dist.is_initialized():
        return tensor

    world_size = dist.get_world_size(group=group)
    if world_size == 1:
        return tensor

    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor, group=group)
    return torch.cat(gathered, dim=0)
