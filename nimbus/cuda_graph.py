"""CUDA graphs for decode steps.

A decode step launches ~400 small kernels; from Python that takes ~30 ms of
CPU time while the GPU needs ~15 ms (measured on TinyLlama, 64 sequences),
so the GPU sits idle half the time. A CUDA graph records the launches once
and replays them with a single call.

Graphs need fixed shapes and fixed memory addresses, so:
- inputs live in static buffers sized for the largest batch, and a step's
  real batch is padded up to the nearest captured size;
- padded rows read and write a scratch KV block that no sequence owns;
- the decode attention grid is fixed per batch size (num_partitions), with
  partitions beyond a sequence's length contributing nothing.
Steps that contain any prefill work run eagerly.
"""

from __future__ import annotations

import bisect

import torch
import triton

from nimbus.kernels.paged_attention import PARTITION
from nimbus.model.attention import AttnMetadata
from nimbus.sequence import Sequence


class DecodeGraphs:
    def __init__(self, model, kv_layers, block_size: int, max_model_len: int, max_batch: int,
                 scratch_block: int, num_kv_heads: int, device):
        self.model = model
        self.kv_layers = kv_layers
        self.block_size = block_size
        self.max_model_len = max_model_len
        self.scratch_slot = scratch_block * block_size
        self.scratch_block = scratch_block
        self.sizes = [b for b in (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128) if b < max_batch] + [max_batch]
        self.max_batch = max_batch
        width = (max_model_len + block_size - 1) // block_size
        self.input_ids = torch.zeros(max_batch, dtype=torch.int64, device=device)
        self.positions = torch.zeros(max_batch, dtype=torch.int64, device=device)
        self.slots = torch.full((max_batch,), self.scratch_slot, dtype=torch.int64, device=device)
        self.ctx = torch.ones(max_batch, dtype=torch.int32, device=device)
        self.tables = torch.full((max_batch, width), scratch_block, dtype=torch.int32, device=device)
        self._host = torch.empty((4, max_batch), dtype=torch.int64, pin_memory=True)
        self._host_tables = torch.empty((max_batch, width), dtype=torch.int32, pin_memory=True)
        self.graphs: dict[int, tuple[torch.cuda.CUDAGraph, torch.Tensor, AttnMetadata]] = {}
        self.kv_heads = num_kv_heads
        self.device = device

    def _meta(self, b: int) -> AttnMetadata:
        empty = torch.zeros(0, dtype=torch.int32, device=self.device)
        parts = triton.cdiv(self.max_model_len, PARTITION) if b * self.kv_heads < 64 else 1
        return AttnMetadata(
            slot_mapping=self.slots[:b], num_prefill_tokens=0,
            prefill_query_start=torch.zeros(1, dtype=torch.int32, device=self.device),
            prefill_context_lens=empty, prefill_block_tables=empty.view(0, 1),
            decode_context_lens=self.ctx[:b], decode_block_tables=self.tables[:b],
            max_decode_context=self.max_model_len,
            prefill_query_start_host=[0], prefill_context_lens_host=[],
            decode_partitions=parts,
        )

    @torch.inference_mode()
    def capture(self) -> None:
        pool = torch.cuda.graph_pool_handle()
        for b in reversed(self.sizes):  # largest first, so the pool is sized once
            meta = self._meta(b)
            args = (self.input_ids[:b], self.positions[:b], self.kv_layers, meta)
            for _ in range(2):  # warm up (Triton compiles, cuBLAS picks algorithms)
                self.model.logits(self.model(*args))
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool):
                logits = self.model.logits(self.model(*args))
            # Keep meta's tensors alive for as long as the graph can replay.
            self.graphs[b] = (g, logits, meta)
        torch.cuda.synchronize()

    def can_run(self, batch: list[tuple[Sequence, int]]) -> bool:
        return 0 < len(batch) <= self.max_batch and all(
            n == 1 and s.num_computed > 0 and s.num_computed + 1 == len(s) for s, n in batch)

    @torch.inference_mode()
    def run(self, seqs: list[Sequence]) -> torch.Tensor:
        """Runs one decode step for seqs; returns their logits [len(seqs), vocab]."""
        g, logits, _ = self.graphs[self._load(seqs)]
        g.replay()
        return logits[: len(seqs)]

    @torch.inference_mode()
    def greedy_chain(self, seqs: list[Sequence], steps: int) -> torch.Tensor:
        """Runs `steps` greedy decode steps without returning to the host:
        each step's argmax becomes the next step's input on the GPU, and
        positions, slots and context lengths advance on the GPU too. The
        caller must have grown every block table by `steps` tokens.
        Returns the chosen tokens [len(seqs), steps]."""
        n = len(seqs)
        b = self._load(seqs)
        g, logits, _ = self.graphs[b]
        rows = torch.arange(b, device=self.device)
        out = []
        for i in range(steps):
            g.replay()
            tok = logits.argmax(-1)  # [b]
            out.append(tok[:n])
            if i + 1 == steps:
                break
            self.input_ids[:b].copy_(tok)
            self.positions[:n] += 1
            self.ctx[:n] += 1
            pos = self.positions[:n]
            page = self.tables[rows[:n], pos // self.block_size].long()
            self.slots[:n] = page * self.block_size + pos % self.block_size
        return torch.stack(out, dim=1)

    def _load(self, seqs: list[Sequence]) -> int:
        """Copies seqs' next-token inputs into the static buffers; returns
        the captured batch size used."""
        n = len(seqs)
        b = self.sizes[bisect.bisect_left(self.sizes, n)]
        bs = self.block_size
        h = self._host
        ht = self._host_tables
        for i, s in enumerate(seqs):
            p = s.num_computed
            h[0, i] = s.token_ids[p]
            h[1, i] = p
            h[2, i] = s.block_table[p // bs] * bs + p % bs
            h[3, i] = p + 1
            ht[i, : len(s.block_table)] = torch.as_tensor(s.block_table, dtype=torch.int32)
        # padding rows point at the scratch block
        h[0, n:b] = 0
        h[1, n:b] = 0
        h[2, n:b] = self.scratch_slot
        h[3, n:b] = 1
        ht[n:b, 0] = self.scratch_block
        self.input_ids[:b].copy_(h[0, :b], non_blocking=True)
        self.positions[:b].copy_(h[1, :b], non_blocking=True)
        self.slots[:b].copy_(h[2, :b], non_blocking=True)
        self.ctx[:b].copy_(h[3, :b], non_blocking=True)
        self.tables[:b].copy_(ht[:b], non_blocking=True)
        return b
