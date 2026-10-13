"""Triton paged attention for decode (one query token per sequence).

One program handles one (sequence, KV head, context partition). It loads
the query heads that share that KV head (grouped-query attention), streams
the partition's keys and values tile by tile straight out of the paged
cache through the block table, and keeps a running max and sum (online
softmax) so the whole score row never has to exist in memory.

Long contexts are split into partitions processed by separate programs
("flash-decoding"); a second step merges their partial results. With many
sequences one partition is enough and the kernel writes the final output
directly.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

PARTITION = 512  # context tokens per program
TILE = 64  # tokens per inner iteration


@triton.jit
def _paged_decode_kernel(
    q_ptr, k_ptr, v_ptr, bt_ptr, ctx_ptr,
    out_ptr, acc_ptr, m_ptr, l_ptr,
    scale,
    stride_qb, stride_qh,
    stride_btb,
    stride_kb, stride_kt, stride_kh,
    stride_ob, stride_oh,
    G: tl.constexpr, G_PAD: tl.constexpr, D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, TILE: tl.constexpr, PARTITION: tl.constexpr,
    NUM_PARTITIONS: tl.constexpr,
):
    b = tl.program_id(0)
    kvh = tl.program_id(1)
    part = tl.program_id(2)
    ctx = tl.load(ctx_ptr + b)
    start = part * PARTITION
    end = tl.minimum(start + PARTITION, ctx)

    g = tl.arange(0, G_PAD)
    d = tl.arange(0, D)
    gmask = g < G
    h = kvh * G + g
    q = tl.load(q_ptr + b * stride_qb + h[:, None] * stride_qh + d[None, :], mask=gmask[:, None], other=0.0)

    m_i = tl.full([G_PAD], float("-inf"), tl.float32)
    l_i = tl.zeros([G_PAD], tl.float32)
    acc = tl.zeros([G_PAD, D], tl.float32)
    for t0 in range(start, end, TILE):
        t = t0 + tl.arange(0, TILE)
        tmask = t < end
        page = tl.load(bt_ptr + b * stride_btb + t // BLOCK_SIZE, mask=tmask, other=0)
        row = page.to(tl.int64) * stride_kb + (t % BLOCK_SIZE) * stride_kt + kvh * stride_kh
        k = tl.load(k_ptr + row[:, None] + d[None, :], mask=tmask[:, None], other=0.0)  # [TILE, D]
        s = tl.dot(q, tl.trans(k)) * scale  # [G_PAD, TILE], fp32
        s = tl.where(tmask[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(v_ptr + row[:, None] + d[None, :], mask=tmask[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    if NUM_PARTITIONS == 1:
        o = acc / l_i[:, None]
        tl.store(out_ptr + b * stride_ob + h[:, None] * stride_oh + d[None, :], o.to(out_ptr.dtype.element_ty),
                 mask=gmask[:, None])
    else:
        # partials laid out [B, H, P, D] and [B, H, P]
        H = tl.num_programs(1) * G
        base = (b * H + h) * NUM_PARTITIONS + part
        tl.store(acc_ptr + base[:, None] * D + d[None, :], acc, mask=gmask[:, None])
        tl.store(m_ptr + base, m_i, mask=gmask)
        tl.store(l_ptr + base, l_i, mask=gmask)


@triton.jit
def _merge_partitions_kernel(acc_ptr, m_ptr, l_ptr, out_ptr, stride_ob, stride_oh, H,
                             D: tl.constexpr, P: tl.constexpr, P_PAD: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    p = tl.arange(0, P_PAD)
    d = tl.arange(0, D)
    base = (b * H + h) * P
    pmask = p < P
    m = tl.load(m_ptr + base + p, mask=pmask, other=float("-inf"))
    l = tl.load(l_ptr + base + p, mask=pmask, other=0.0)
    m_max = tl.max(m, 0)
    w = tl.where(pmask & (m > float("-inf")), tl.exp(m - m_max), 0.0)  # empty partitions weigh 0
    acc = tl.load(acc_ptr + (base + p)[:, None] * D + d[None, :], mask=pmask[:, None], other=0.0)
    o = tl.sum(acc * w[:, None], 0) / tl.sum(l * w, 0)
    tl.store(out_ptr + b * stride_ob + h * stride_oh + d, o.to(out_ptr.dtype.element_ty))


def paged_decode_attention(
    q: torch.Tensor,  # [B, H, D]
    k_cache: torch.Tensor,  # [num_blocks, block_size, KV, D]
    v_cache: torch.Tensor,
    context_lens: torch.Tensor,  # [B] int32
    block_tables: torch.Tensor,  # [B, max_blocks] int32
    max_context: int,
    scale: float,
    num_partitions: int | None = None,
) -> torch.Tensor:
    """num_partitions fixes the launch grid regardless of the actual context
    lengths, which CUDA graph capture needs; partitions past a sequence's
    end are empty and drop out in the merge."""
    b, h, d = q.shape
    kv = k_cache.shape[2]
    g = h // kv
    out = torch.empty_like(q)
    if b == 0:
        return out
    # Split the context only when there is too little parallelism otherwise.
    if num_partitions is None:
        num_partitions = triton.cdiv(max_context, PARTITION) if b * kv < 64 else 1
    parts = num_partitions
    if parts > 1:
        acc = torch.empty((b, h, parts, d), dtype=torch.float32, device=q.device)
        m = torch.empty((b, h, parts), dtype=torch.float32, device=q.device)
        l = torch.empty_like(m)
    else:
        acc = m = l = out  # unused
    partition = PARTITION if parts > 1 else 1 << 30  # one partition covers everything
    _paged_decode_kernel[(b, kv, parts)](
        q, k_cache, v_cache, block_tables, context_lens,
        out, acc, m, l,
        scale,
        q.stride(0), q.stride(1),
        block_tables.stride(0),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        out.stride(0), out.stride(1),
        G=g, G_PAD=max(16, triton.next_power_of_2(g)), D=d,
        BLOCK_SIZE=k_cache.shape[1], TILE=TILE, PARTITION=partition, NUM_PARTITIONS=parts,
        num_warps=4,
    )
    if parts > 1:
        _merge_partitions_kernel[(b, h)](acc, m, l, out, out.stride(0), out.stride(1), h,
                                         D=d, P=parts, P_PAD=max(2, triton.next_power_of_2(parts)))
    return out
