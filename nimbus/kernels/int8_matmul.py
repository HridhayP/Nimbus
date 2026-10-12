"""Triton matmul with INT8 weights: y = x @ (w_q * scale)^T.

The weight tile is loaded as int8 (half the bytes of fp16), converted to
fp16 in registers, multiplied on tensor cores with fp32 accumulation, and
the per-output-channel scale is applied once in the epilogue. Decode is
bound by reading weights from memory, so halving their size is the point.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def _configs():
    out = []
    for bm, bn, bk, w, s in [
        (16, 64, 128, 4, 4), (16, 128, 64, 4, 4), (32, 64, 128, 4, 4), (32, 128, 64, 4, 3),
        (64, 128, 64, 4, 3), (128, 128, 64, 8, 3), (64, 256, 32, 8, 3),
    ]:
        out.append(triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk}, num_warps=w, num_stages=s))
    return out


@triton.autotune(configs=_configs(), key=["M_BUCKET", "N", "K"])
@triton.jit
def _int8_matmul_kernel(
    x_ptr, w_ptr, s_ptr, y_ptr, M, N, K, M_BUCKET,
    stride_xm, stride_wn, stride_ym,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + rk
        x = tl.load(x_ptr + rm[:, None] * stride_xm + kk[None, :],
                    mask=(rm[:, None] < M) & (kk[None, :] < K), other=0.0)
        w = tl.load(w_ptr + rn[None, :] * stride_wn + kk[:, None],
                    mask=(rn[None, :] < N) & (kk[:, None] < K), other=0)  # [BLOCK_K, BLOCK_N] int8
        acc += tl.dot(x, w.to(x.dtype))
    scale = tl.load(s_ptr + rn, mask=rn < N, other=0.0).to(tl.float32)
    y = (acc * scale[None, :]).to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + rm[:, None] * stride_ym + rn[None, :], y, mask=(rm[:, None] < M) & (rn[None, :] < N))


def int8_matmul(x: torch.Tensor, w_q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """x: [..., K] fp16; w_q: [N, K] int8; scale: [N]. Returns [..., N]."""
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    m, k = x2.shape
    n = w_q.shape[0]
    y = torch.empty((m, n), dtype=x.dtype, device=x.device)
    # Autotune per size class of M, not per exact M (decode batches vary).
    m_bucket = triton.next_power_of_2(max(m, 16)) if m <= 256 else 512
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]), triton.cdiv(n, meta["BLOCK_N"]))  # noqa: E731
    _int8_matmul_kernel[grid](x2, w_q, scale, y, m, n, k, m_bucket, x2.stride(0), w_q.stride(0), y.stride(0))
    return y.view(*shape[:-1], n)
