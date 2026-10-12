"""Triton RMSNorm, optionally fused with the residual add before it.

Unfused, a decoder layer's "residual += x; h = rms_norm(residual)" reads
and writes the hidden state several times across separate kernels (add,
cast, square, mean, rsqrt, multiply, cast, multiply). Here one program
reads a row once, does the add and the normalization in registers, and
writes the new residual and the normalized row.

Numerics follow HuggingFace's LlamaRMSNorm: the add happens in the model
dtype, the normalization in float32, then the result is cast back and
scaled by the weight.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(x_ptr, res_ptr, w_ptr, out_ptr, res_out_ptr, n_cols, eps,
                    stride_x, stride_r, stride_o, HAS_RESIDUAL: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0)
    if HAS_RESIDUAL:
        r = tl.load(res_ptr + row * stride_r + cols, mask=mask, other=0.0)
        x = x + r  # model dtype, like HF's residual add
        tl.store(res_out_ptr + row * stride_r + cols, x, mask=mask)
    xf = x.to(tl.float32)
    var = tl.sum(xf * xf, axis=0) / n_cols
    y = (xf * tl.rsqrt(var + eps)).to(x.dtype)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    tl.store(out_ptr + row * stride_o + cols, (w * y).to(out_ptr.dtype.element_ty), mask=mask)


def _launch(x, residual, weight, eps):
    x2 = x.reshape(-1, x.shape[-1])
    out = torch.empty_like(x2)
    res_out = torch.empty_like(x2) if residual is not None else out
    r2 = residual.reshape(-1, x.shape[-1]) if residual is not None else x2
    n = x2.shape[1]
    _rmsnorm_kernel[(x2.shape[0],)](
        x2, r2, weight, out, res_out, n, eps,
        x2.stride(0), r2.stride(0), out.stride(0),
        HAS_RESIDUAL=residual is not None, BLOCK=triton.next_power_of_2(n), num_warps=8,
    )
    return out.view_as(x), res_out.view_as(x)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return _launch(x, None, weight, eps)[0]


def add_rms_norm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float):
    """residual += x; returns (rms_norm(residual), residual). Does not modify inputs."""
    return _launch(x, residual, weight, eps)
