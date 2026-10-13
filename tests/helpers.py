"""Test helpers: tiny random Llama checkpoints and a minimal one-sequence
greedy loop that drives the model directly (no scheduler)."""

from __future__ import annotations

import os

import torch

from nimbus.cache.kv_cache import KVCache
from nimbus.model.attention import AttnMetadata

TINYLLAMA = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def make_tiny_llama(path: str, *, kv_heads: int = 2, layers: int = 2, seed: int = 0, rope_scaling=None) -> str:
    """Saves a small random HF LlamaForCausalLM to path and returns path."""
    from transformers import LlamaConfig, LlamaForCausalLM

    if os.path.exists(os.path.join(path, "config.json")):
        return path
    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=256, hidden_size=64, intermediate_size=176, num_hidden_layers=layers,
        num_attention_heads=4, num_key_value_heads=kv_heads, max_position_embeddings=256,
        rms_norm_eps=1e-5, tie_word_embeddings=False, rope_scaling=rope_scaling,
    )
    model = LlamaForCausalLM(cfg)
    # Random init gives nearly flat logits; scale up so argmax is decisive.
    with torch.no_grad():
        model.lm_head.weight.mul_(20)
    model.save_pretrained(path, safe_serialization=True)
    return path


def single_seq_meta(ctx_len: int, q_len: int, block_size: int, device) -> AttnMetadata:
    """Metadata for one sequence whose blocks are 0, 1, 2, ... in order."""
    nb = (ctx_len + block_size - 1) // block_size
    table = torch.arange(nb, dtype=torch.int32, device=device)[None, :]
    slots = torch.arange(ctx_len - q_len, ctx_len, dtype=torch.int64, device=device)
    empty_i = torch.zeros(0, dtype=torch.int32, device=device)
    if q_len > 1 or ctx_len == q_len:
        return AttnMetadata(
            slot_mapping=slots, num_prefill_tokens=q_len,
            prefill_query_start=torch.tensor([0, q_len], dtype=torch.int32, device=device),
            prefill_context_lens=torch.tensor([ctx_len], dtype=torch.int32, device=device),
            prefill_block_tables=table,
            decode_context_lens=empty_i, decode_block_tables=empty_i.view(0, 1), max_decode_context=0,
            prefill_query_start_host=[0, q_len], prefill_context_lens_host=[ctx_len],
        )
    return AttnMetadata(
        slot_mapping=slots, num_prefill_tokens=0,
        prefill_query_start=torch.zeros(1, dtype=torch.int32, device=device),
        prefill_context_lens=empty_i, prefill_block_tables=empty_i.view(0, 1),
        decode_context_lens=torch.tensor([ctx_len], dtype=torch.int32, device=device),
        decode_block_tables=table, max_decode_context=ctx_len,
        prefill_query_start_host=[0], prefill_context_lens_host=[],
    )


@torch.inference_mode()
def greedy_generate(model, cfg, prompt: list[int], n: int, device, block_size: int = 16):
    """Returns (generated token ids, list of per-step logits)."""
    dtype = model.embed_tokens.weight.dtype
    cache = KVCache(cfg, (len(prompt) + n) // block_size + 2, block_size, dtype, device)
    ids = torch.tensor(prompt, device=device)
    pos = torch.arange(len(prompt), device=device)
    meta = single_seq_meta(len(prompt), len(prompt), block_size, device)
    h = model(ids, pos, cache.layers, meta)
    logits = model.logits(h[-1:])
    out, all_logits = [], [logits[0]]
    for i in range(n):
        tok = int(logits[0].argmax())
        out.append(tok)
        if i == n - 1:
            break
        ctx = len(prompt) + i + 1
        meta = single_seq_meta(ctx, 1, block_size, device)
        h = model(torch.tensor([tok], device=device), torch.tensor([ctx - 1], device=device), cache.layers, meta)
        logits = model.logits(h)
        all_logits.append(logits[0])
    return out, all_logits
