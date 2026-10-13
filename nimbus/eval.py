"""Perplexity evaluation with the sliding-window method used by
HuggingFace's perplexity guide: windows of `window` tokens advance by
`stride`, and each window scores only the tokens not scored by the
previous one, so every token is predicted with up to window-stride tokens
of context."""

from __future__ import annotations

import math

import torch

from nimbus.cache.kv_cache import KVCache
from nimbus.config import ModelConfig
from nimbus.runner import prepare
from nimbus.sequence import SamplingParams, Sequence


@torch.inference_mode()
def perplexity(model, cfg: ModelConfig, ids: list[int], window: int = 2048, stride: int = 512,
               block_size: int = 16) -> float:
    device = model.embed_tokens.weight.device
    dtype = model.embed_tokens.weight.dtype
    nb = (window + block_size - 1) // block_size
    cache = KVCache(cfg, nb, block_size, dtype, device)
    nll, count, prev_end = 0.0, 0, 0
    for begin in range(0, len(ids), stride):
        end = min(begin + window, len(ids))
        chunk = ids[begin:end]
        seq = Sequence(0, chunk, SamplingParams())
        seq.block_table = list(range((len(chunk) + block_size - 1) // block_size))
        step = prepare([(seq, len(chunk))], block_size, device)
        h = model(step.input_ids, step.positions, cache.layers, step.meta)
        new = end - prev_end  # tokens not yet scored
        # position j's logits predict token j+1; score the last `new` targets
        first = max(len(chunk) - new, 1)
        logits = model.logits(h[first - 1: len(chunk) - 1])
        targets = torch.tensor(chunk[first:], device=device)
        nll += torch.nn.functional.cross_entropy(logits, targets, reduction="sum").item()
        count += len(chunk) - first
        prev_end = end
        if end == len(ids):
            break
    return math.exp(nll / count)
