"""PyTorch reference implementations.

These run everywhere (CPU in CI, GPU without Triton) and are the oracles the
Triton kernels are tested against. Numerics follow HuggingFace's Llama code
so that greedy output matches it token for token.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def write_kv(
    k: torch.Tensor,  # [T, KV, D]
    v: torch.Tensor,
    k_cache: torch.Tensor,  # [num_blocks, block_size, KV, D]
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,  # [T] int64
) -> None:
    kv, d = k_cache.shape[2], k_cache.shape[3]
    k_cache.view(-1, kv, d).index_copy_(0, slot_mapping, k)
    v_cache.view(-1, kv, d).index_copy_(0, slot_mapping, v)


def gather_kv(
    cache: torch.Tensor,  # [num_blocks, block_size, KV, D]
    block_table: torch.Tensor,  # [max_blocks]
    length: int,
) -> torch.Tensor:
    """Returns the first ``length`` cached tokens of one sequence: [length, KV, D]."""
    bs = cache.shape[1]
    nb = (length + bs - 1) // bs
    return cache[block_table[:nb].long()].flatten(0, 1)[:length]


def paged_prefill_attention(
    q: torch.Tensor,  # [Tp, H, D]
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    query_start: torch.Tensor,  # [P+1]
    context_lens: torch.Tensor,  # [P]
    block_tables: torch.Tensor,  # [P, max_blocks]
    scale: float,
    query_start_host: list[int],
    context_lens_host: list[int],
) -> torch.Tensor:
    """Causal attention for sequences whose newest q_len tokens are queries.

    Query j of a sequence with context length c sits at absolute position
    c - q_len + j and may attend to keys 0..that position.
    """
    out = torch.empty_like(q)
    for i, ctx in enumerate(context_lens_host):
        s, e = query_start_host[i], query_start_host[i + 1]
        qi = q[s:e].transpose(0, 1).unsqueeze(0)  # [1, H, q_len, D]
        ki = gather_kv(k_cache, block_tables[i], ctx).transpose(0, 1).unsqueeze(0)
        vi = gather_kv(v_cache, block_tables[i], ctx).transpose(0, 1).unsqueeze(0)
        q_len = e - s
        if q_len == ctx:
            o = F.scaled_dot_product_attention(qi, ki, vi, is_causal=True, scale=scale, enable_gqa=True)
        else:
            qpos = torch.arange(ctx - q_len, ctx, device=q.device)[:, None]
            kpos = torch.arange(ctx, device=q.device)[None, :]
            o = F.scaled_dot_product_attention(qi, ki, vi, attn_mask=kpos <= qpos, scale=scale, enable_gqa=True)
        out[s:e] = o[0].transpose(0, 1)
    return out


def paged_decode_attention(
    q: torch.Tensor,  # [B, H, D], one query per sequence
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    context_lens: torch.Tensor,  # [B]
    block_tables: torch.Tensor,  # [B, max_blocks]
    max_context: int,
    scale: float,
) -> torch.Tensor:
    b = q.shape[0]
    bs = k_cache.shape[1]
    nb = (max_context + bs - 1) // bs
    bt = block_tables[:, :nb].long()
    k = k_cache[bt].flatten(1, 2)  # [B, nb*bs, KV, D]
    v = v_cache[bt].flatten(1, 2)
    s = k.shape[1]
    mask = torch.arange(s, device=q.device)[None, :] < context_lens[:, None]  # [B, S]
    o = F.scaled_dot_product_attention(
        q.unsqueeze(2),  # [B, H, 1, D]
        k.transpose(1, 2),  # [B, KV, S, D]
        v.transpose(1, 2),
        attn_mask=mask[:, None, None, :],
        scale=scale,
        enable_gqa=True,
    )
    return o.view(b, q.shape[1], q.shape[2])


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    # Same order of operations as HuggingFace LlamaRMSNorm.
    dtype = x.dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return weight * xf.to(dtype)


def add_rms_norm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """residual += x; returns (rms_norm(residual), residual)."""
    residual = residual + x
    return rms_norm(residual, weight, eps), residual
