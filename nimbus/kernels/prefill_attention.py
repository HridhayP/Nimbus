"""Triton causal attention for prefill chunks over the paged cache.

FlashAttention-style: one program owns a tile of BLOCK_M queries of one
sequence and one head, and streams that sequence's keys and values in
tiles of BLOCK_N straight from the paged cache, keeping a running max and
sum. The scores never leave registers and the cache is never copied into
a contiguous buffer.

A sequence's queries are its newest q_len tokens; query j sits at absolute
position ctx - q_len + j. This covers ordinary prompts (q_len == ctx),
chunked prefill continuing a cached prefix, and speculative verification
of several draft tokens at once.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

BLOCK_M = 64
BLOCK_N = 64


@triton.jit
def _paged_prefill_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr, bt_ptr,
    prog_seq_ptr, prog_m_ptr, qstart_ptr, ctx_ptr,
    scale,
    stride_qt, stride_qh,
    stride_btb,
    stride_kb, stride_kt, stride_kh,
    G: tl.constexpr, D: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    h = tl.program_id(1)
    kvh = h // G
    seq = tl.load(prog_seq_ptr + pid)
    m0 = tl.load(prog_m_ptr + pid)
    qs = tl.load(qstart_ptr + seq)
    q_len = tl.load(qstart_ptr + seq + 1) - qs
    ctx = tl.load(ctx_ptr + seq)

    offs_m = m0 + tl.arange(0, BLOCK_M)
    mmask = offs_m < q_len
    qpos = ctx - q_len + offs_m  # absolute positions of the queries
    d = tl.arange(0, D)
    q = tl.load(q_ptr + (qs + offs_m)[:, None] * stride_qt + h * stride_qh + d[None, :],
                mask=mmask[:, None], other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    # Keys past the last query of this tile are masked by causality anyway.
    kv_end = tl.minimum(ctx, ctx - q_len + m0 + BLOCK_M)
    for n0 in range(0, kv_end, BLOCK_N):
        t = n0 + tl.arange(0, BLOCK_N)
        tmask = t < kv_end
        page = tl.load(bt_ptr + seq * stride_btb + t // BLOCK_SIZE, mask=tmask, other=0)
        row = page.to(tl.int64) * stride_kb + (t % BLOCK_SIZE) * stride_kt + kvh * stride_kh
        k = tl.load(k_ptr + row[:, None] + d[None, :], mask=tmask[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale  # [BLOCK_M, BLOCK_N]
        allowed = (t[None, :] <= qpos[:, None]) & tmask[None, :]
        s = tl.where(allowed, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        # Rows that are fully masked so far keep m = -inf; guard the exps.
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp(s - m_safe[:, None])
        alpha = tl.exp(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(v_ptr + row[:, None] + d[None, :], mask=tmask[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    o = acc / tl.where(l_i == 0, 1.0, l_i)[:, None]
    tl.store(out_ptr + (qs + offs_m)[:, None] * stride_qt + h * stride_qh + d[None, :],
             o.to(out_ptr.dtype.element_ty), mask=mmask[:, None])


def build_programs(query_start_host: list[int], device) -> tuple[torch.Tensor, torch.Tensor]:
    """(sequence index, first query offset) for every BLOCK_M query tile."""
    seqs, starts = [], []
    for i in range(len(query_start_host) - 1):
        q_len = query_start_host[i + 1] - query_start_host[i]
        for m0 in range(0, q_len, BLOCK_M):
            seqs.append(i)
            starts.append(m0)
    t = torch.tensor([seqs, starts], dtype=torch.int32).to(device, non_blocking=True)
    return t[0], t[1]


def paged_prefill_attention(
    q: torch.Tensor,  # [Tp, H, D]
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    query_start: torch.Tensor,  # [P+1] int32
    context_lens: torch.Tensor,  # [P] int32
    block_tables: torch.Tensor,  # [P, max_blocks] int32
    scale: float,
    query_start_host: list[int],
    context_lens_host: list[int],
    programs: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    t, h, d = q.shape
    out = torch.empty_like(q)
    prog_seq, prog_m = programs if programs is not None else build_programs(query_start_host, q.device)
    _paged_prefill_kernel[(prog_seq.shape[0], h)](
        q, k_cache, v_cache, out, block_tables,
        prog_seq, prog_m, query_start, context_lens,
        scale,
        q.stride(0), q.stride(1),
        block_tables.stride(0),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        G=h // k_cache.shape[2], D=d, BLOCK_SIZE=k_cache.shape[1],
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2,
    )
    return out
