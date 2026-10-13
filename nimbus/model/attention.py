"""Attention over a paged KV cache.

A forward pass processes a flat batch of tokens from many sequences. The
first ``num_prefill_tokens`` tokens belong to prefill sequences (several
tokens each, possibly continuing a prefix that is already cached: chunked
prefill and speculative verification both look like this); the rest are
decode tokens, exactly one per sequence.

Every token's key and value are first written into the cache at
``slot_mapping[t]`` (slot = physical block * block_size + offset). Attention
then reads each sequence's keys and values back through its block table, so
the model never needs a sequence's cache to be contiguous.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from nimbus.kernels import reference


@dataclass
class AttnMetadata:
    slot_mapping: torch.Tensor  # [T] int64
    num_prefill_tokens: int
    # Prefill sequences (P of them)
    prefill_query_start: torch.Tensor  # [P+1] int32, offsets into the flat batch
    prefill_context_lens: torch.Tensor  # [P] int32, cached + new tokens
    prefill_block_tables: torch.Tensor  # [P, max_blocks] int32
    # Decode sequences (D of them)
    decode_context_lens: torch.Tensor  # [D] int32, including the new token
    decode_block_tables: torch.Tensor  # [D, max_blocks] int32
    max_decode_context: int
    # Host copies, so the reference path doesn't sync with the GPU
    prefill_query_start_host: list[int]
    prefill_context_lens_host: list[int]
    # Built lazily by the Triton backend, once per step instead of per layer
    prefill_programs: tuple[torch.Tensor, torch.Tensor] | None = None
    # Fixed decode launch grid (set when capturing CUDA graphs)
    decode_partitions: int | None = None

    @property
    def num_prefills(self) -> int:
        return len(self.prefill_context_lens_host)

    @property
    def num_decodes(self) -> int:
        return int(self.decode_context_lens.shape[0])


class AttentionBackend:
    """Chooses between Triton kernels and the PyTorch reference."""

    def __init__(self, use_triton: bool):
        self.use_triton = use_triton
        if use_triton:
            from nimbus.kernels import paged_attention, prefill_attention

            self._decode = paged_attention.paged_decode_attention
            self._prefill = prefill_attention.paged_prefill_attention
            self._build_programs = prefill_attention.build_programs
        else:
            self._decode = reference.paged_decode_attention
            self._prefill = reference.paged_prefill_attention

    def forward(
        self,
        q: torch.Tensor,  # [T, H, D]
        k: torch.Tensor,  # [T, KV, D]
        v: torch.Tensor,
        k_cache: torch.Tensor,  # [num_blocks, block_size, KV, D]
        v_cache: torch.Tensor,
        meta: AttnMetadata,
        scale: float,
    ) -> torch.Tensor:
        reference.write_kv(k, v, k_cache, v_cache, meta.slot_mapping)
        out = torch.empty_like(q)
        p = meta.num_prefill_tokens
        if p:
            if self.use_triton and meta.prefill_programs is None:
                meta.prefill_programs = self._build_programs(meta.prefill_query_start_host, q.device)
            out[:p] = self._prefill(
                q[:p], k_cache, v_cache, meta.prefill_query_start, meta.prefill_context_lens,
                meta.prefill_block_tables, scale, meta.prefill_query_start_host,
                meta.prefill_context_lens_host, meta.prefill_programs,
            )
        if meta.num_decodes:
            kw = {"num_partitions": meta.decode_partitions} if self.use_triton else {}
            out[p:] = self._decode(
                q[p:], k_cache, v_cache, meta.decode_context_lens, meta.decode_block_tables,
                meta.max_decode_context, scale, **kw,
            )
        return out
