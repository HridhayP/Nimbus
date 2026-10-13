"""Preallocated, paged KV cache storage."""

from __future__ import annotations

import torch

from nimbus.config import ModelConfig


class KVCache:
    """One key and one value tensor per layer, each
    [num_blocks, block_size, num_kv_heads, head_dim], allocated once.

    Which sequence owns which block is tracked by the BlockAllocator and
    each sequence's block table; this class only holds the memory.
    """

    def __init__(self, cfg: ModelConfig, num_blocks: int, block_size: int, dtype: torch.dtype, device: str):
        self.num_blocks = num_blocks
        self.block_size = block_size
        shape = (num_blocks, block_size, cfg.num_kv_heads, cfg.head_dim)
        self.layers = [
            (torch.zeros(shape, dtype=dtype, device=device), torch.zeros(shape, dtype=dtype, device=device))
            for _ in range(cfg.num_layers)
        ]

    @staticmethod
    def bytes_per_block(cfg: ModelConfig, block_size: int, dtype: torch.dtype) -> int:
        return block_size * cfg.kv_bytes_per_token(torch.finfo(dtype).bits // 8)

    @property
    def num_bytes(self) -> int:
        k, v = self.layers[0]
        return len(self.layers) * (k.numel() + v.numel()) * k.element_size()
