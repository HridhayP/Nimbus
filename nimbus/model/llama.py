"""Llama-family decoder (TinyLlama, Llama 2/3) over a flat, paged batch.

Projections are fused where it saves kernel launches: q/k/v are one matmul
and gate/up are one matmul. Numerics (RoPE in float32, RMSNorm in float32,
everything else in the model dtype) follow HuggingFace's implementation.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from nimbus.config import ModelConfig
from nimbus.kernels import reference
from nimbus.model.attention import AttentionBackend, AttnMetadata


def rope_inv_freq(cfg: ModelConfig) -> torch.Tensor:
    d = cfg.head_dim
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, d, 2, dtype=torch.int64).float() / d))
    sc = cfg.rope_scaling
    if sc is None:
        return inv_freq
    # Llama 3.1 scaling: long wavelengths are slowed by `factor`, short ones
    # kept, and the band in between interpolated.
    low_wavelen = sc.original_max_position / sc.low_freq_factor
    high_wavelen = sc.original_max_position / sc.high_freq_factor
    wavelen = 2 * math.pi / inv_freq
    scaled = torch.where(wavelen > low_wavelen, inv_freq / sc.factor, inv_freq)
    smooth = (sc.original_max_position / wavelen - sc.low_freq_factor) / (sc.high_freq_factor - sc.low_freq_factor)
    smoothed = (1 - smooth) * scaled / sc.factor + smooth * scaled
    medium = ~(wavelen < high_wavelen) & ~(wavelen > low_wavelen)
    return torch.where(medium, smoothed, scaled)


class Rotary(nn.Module):
    def __init__(self, cfg: ModelConfig, max_position: int):
        super().__init__()
        inv_freq = rope_inv_freq(cfg)
        pos = torch.arange(max_position, dtype=torch.float32)
        freqs = pos[:, None] * inv_freq[None, :]
        emb = torch.cat([freqs, freqs], dim=-1)  # [max_pos, head_dim]
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor):
        # q: [T, H, D], k: [T, KV, D]
        cos = self.cos[positions].to(q.dtype)[:, None, :]
        sin = self.sin[positions].to(q.dtype)[:, None, :]
        return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float, fused=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim))
        self.eps = eps
        self.fused = fused  # optional Triton kernels: (rms_norm, add_rms_norm)

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        if self.fused is not None:
            norm, add_norm = self.fused
        else:
            norm, add_norm = reference.rms_norm, reference.add_rms_norm
        if residual is None:
            return norm(x, self.weight, self.eps)
        return add_norm(x, residual, self.weight, self.eps)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, backend: AttentionBackend):
        super().__init__()
        self.h, self.kv, self.d = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        self.qkv_proj = nn.Linear(cfg.hidden_size, (self.h + 2 * self.kv) * self.d, bias=False)
        self.o_proj = nn.Linear(self.h * self.d, cfg.hidden_size, bias=False)
        self.scale = self.d ** -0.5
        self.backend = backend

    def forward(self, x, positions, rotary, k_cache, v_cache, meta: AttnMetadata):
        t = x.shape[0]
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split([self.h * self.d, self.kv * self.d, self.kv * self.d], dim=-1)
        q = q.view(t, self.h, self.d)
        k = k.view(t, self.kv, self.d)
        v = v.view(t, self.kv, self.d)
        q, k = rotary(q, k, positions)
        o = self.backend.forward(q, k, v.contiguous(), k_cache, v_cache, meta, self.scale)
        return self.o_proj(o.view(t, self.h * self.d))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.inter = cfg.intermediate_size
        self.gate_up_proj = nn.Linear(cfg.hidden_size, 2 * self.inter, bias=False)
        self.down_proj = nn.Linear(self.inter, cfg.hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).split(self.inter, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, backend: AttentionBackend, fused_norm=None):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, fused_norm)
        self.self_attn = Attention(cfg, backend)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, fused_norm)
        self.mlp = MLP(cfg)

    def forward(self, x, residual, positions, rotary, k_cache, v_cache, meta):
        if residual is None:
            residual = x
            h = self.input_layernorm(x)
        else:
            h, residual = self.input_layernorm(x, residual)
        h = self.self_attn(h, positions, rotary, k_cache, v_cache, meta)
        h, residual = self.post_attention_layernorm(h, residual)
        return self.mlp(h), residual


class LlamaModel(nn.Module):
    def __init__(self, cfg: ModelConfig, backend: AttentionBackend, max_position: int | None = None, fused_norm=None):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg, backend, fused_norm) for _ in range(cfg.num_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, fused_norm)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.rotary = Rotary(cfg, max_position or cfg.max_position)

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,  # [T]
        positions: torch.Tensor,  # [T]
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]],
        meta: AttnMetadata,
    ) -> torch.Tensor:
        """Returns final hidden states [T, hidden]."""
        x = self.embed_tokens(input_ids)
        residual = None
        for layer, (kc, vc) in zip(self.layers, kv_caches):
            x, residual = layer(x, residual, positions, self.rotary, kc, vc, meta)
        x, _ = self.norm(x, residual)
        return x

    @torch.inference_mode()
    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden).float()
