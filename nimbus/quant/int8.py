"""INT8 weight-only quantization (symmetric, per output channel).

Each row of a weight matrix gets its own scale, max|row| / 127, and is
stored as round(w / scale) in int8. Activations stay fp16. Because the
scale is per output channel, x @ (w_q * s)^T == (x @ w_q^T) * s, so the
scale can be applied after the matmul instead of dequantizing the weight.

The projections inside decoder layers are quantized; the token embedding
and the output head stay in fp16, which keeps the vocabulary-sized
matrices exact (a common choice that protects quality at little cost).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def quantize_weight(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """w: [N, K] float -> (int8 [N, K], scale [N] in w's dtype)."""
    wf = w.float()
    scale = wf.abs().amax(dim=1).clamp_min(1e-8) / 127.0
    q = torch.round(wf / scale[:, None]).clamp_(-127, 127).to(torch.int8)
    return q, scale.to(w.dtype)


def dequantize_weight(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.to(scale.dtype) * scale[:, None]


class Int8Linear(nn.Module):
    def __init__(self, linear: nn.Linear, use_triton: bool):
        super().__init__()
        q, s = quantize_weight(linear.weight.data)
        self.register_buffer("weight_q", q)
        self.register_buffer("scale", s)
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.use_triton = use_triton

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_triton:
            from nimbus.kernels.int8_matmul import int8_matmul

            return int8_matmul(x, self.weight_q, self.scale)
        # Reference: integer weights promoted to the activation dtype, scale after.
        return F.linear(x, self.weight_q.to(x.dtype)) * self.scale.to(x.dtype)


QUANTIZED = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")


@torch.no_grad()
def quantize_model(model: nn.Module, use_triton: bool) -> nn.Module:
    """Replaces decoder projections with Int8Linear in place, freeing the fp16 weights."""
    for layer in model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for name in QUANTIZED:
                lin = getattr(parent, name, None)
                if isinstance(lin, nn.Linear):
                    setattr(parent, name, Int8Linear(lin, use_triton))
                    del lin
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model


def weight_bytes(model: nn.Module) -> int:
    return sum(t.numel() * t.element_size() for t in list(model.parameters()) + list(model.buffers())
               if t.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.int8)
               and not t.is_meta)
