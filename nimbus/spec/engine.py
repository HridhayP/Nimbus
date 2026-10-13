"""Speculative decoding: a small draft model proposes, the target verifies.

Each round, for every active sequence:

1. The draft model runs k decode steps, proposing d_1..d_k (greedy, or
   sampled from its distributions q_i).
2. The target model runs ONE forward pass over [pending token, d_1..d_k]
   (k+1 query tokens attending to the cached prefix, the same shape as a
   chunked-prefill step) and produces distributions p_1..p_{k+1}.
3. Acceptance (spec/accept.py) keeps a prefix of the drafts and adds one
   token from the target, so a round emits 1..k+1 tokens for the cost of
   k draft steps plus one target pass.
4. Rollback is free: rejected positions in either cache are simply marked
   uncomputed and overwritten next round, and blocks past the new length
   are released.

Both the draft's decode steps and the target's verification are
fixed-shape, so both run as CUDA graphs on the GPU path. This engine runs
a static batch to completion (no continuous admission); it exists to
measure acceptance rate and speedup.
"""

from __future__ import annotations

import bisect
import time
from dataclasses import dataclass

import torch

from nimbus.cache.block_manager import BlockManager
from nimbus.cache.kv_cache import KVCache
from nimbus.config import EngineConfig
from nimbus.engine import LLMEngine
from nimbus.model.attention import AttentionBackend, AttnMetadata
from nimbus.model.weights import load_model
from nimbus.runner import prepare
from nimbus.sequence import SamplingParams, Sequence, SeqStatus
from nimbus.spec.accept import accept_greedy, accept_sampled


@dataclass
class SpecStats:
    rounds: int = 0
    proposed: int = 0
    accepted: int = 0
    emitted: int = 0
    draft_time: float = 0.0
    verify_time: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def tokens_per_round(self) -> float:
        return self.emitted / self.rounds if self.rounds else 0.0


class VerifyGraphs:
    """CUDA graphs for the target's verification pass: every sequence feeds
    exactly k+1 tokens. Padding sequences live in the scratch block, which
    holds k+1 <= block_size tokens."""

    def __init__(self, model, kv_layers, block_size, max_model_len, max_batch, k, scratch_block, device):
        assert k + 1 <= block_size
        self.model, self.kv_layers, self.bs, self.q = model, kv_layers, block_size, k + 1
        self.scratch_block = scratch_block
        self.sizes = [b for b in (1, 2, 4, 8, 16) if b < max_batch] + [max_batch]
        mb, q = max_batch, k + 1
        width = (max_model_len + block_size - 1) // block_size
        self.input_ids = torch.zeros(mb * q, dtype=torch.int64, device=device)
        self.positions = torch.zeros(mb * q, dtype=torch.int64, device=device)
        self.slots = torch.zeros(mb * q, dtype=torch.int64, device=device)
        self.ctx = torch.full((mb,), q, dtype=torch.int32, device=device)
        self.tables = torch.full((mb, width), scratch_block, dtype=torch.int32, device=device)
        self.qstart = torch.arange(0, (mb + 1) * q, q, dtype=torch.int32, device=device)
        self._host = torch.empty((3, mb * q), dtype=torch.int64, pin_memory=True)
        self._host_ctx = torch.empty(mb, dtype=torch.int32, pin_memory=True)
        self._host_tables = torch.empty((mb, width), dtype=torch.int32, pin_memory=True)
        self.device = device
        self.graphs = {}

    def _meta(self, b):
        empty = torch.zeros(0, dtype=torch.int32, device=self.device)
        progs = (torch.arange(b, dtype=torch.int32, device=self.device),
                 torch.zeros(b, dtype=torch.int32, device=self.device))
        return AttnMetadata(
            slot_mapping=self.slots[: b * self.q], num_prefill_tokens=b * self.q,
            prefill_query_start=self.qstart[: b + 1], prefill_context_lens=self.ctx[:b],
            prefill_block_tables=self.tables[:b],
            decode_context_lens=empty, decode_block_tables=empty.view(0, 1), max_decode_context=0,
            prefill_query_start_host=list(range(0, (b + 1) * self.q, self.q)),
            prefill_context_lens_host=[self.q] * b, prefill_programs=progs,
        )

    @torch.inference_mode()
    def capture(self):
        pool = torch.cuda.graph_pool_handle()
        for b in reversed(self.sizes):
            n = b * self.q
            # Fill the static buffers with valid padding before warming up.
            self._fill([], b)
            meta = self._meta(b)
            args = (self.input_ids[:n], self.positions[:n], self.kv_layers, meta)
            for _ in range(2):
                self.model.logits(self.model(*args))
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool):
                logits = self.model.logits(self.model(*args))
            # The graph reads tensors owned by meta (e.g. the prefill tile
            # table); keep it alive or replay reads freed memory.
            self.graphs[b] = (g, logits, meta)
        torch.cuda.synchronize()

    def _fill(self, seqs, b):
        q, bs, h, hc, ht = self.q, self.bs, self._host, self._host_ctx, self._host_tables
        for i in range(b):
            if i < len(seqs):
                s = seqs[i]
                start = len(s) - q
                for j in range(q):
                    p = start + j
                    h[0, i * q + j] = s.token_ids[p]
                    h[1, i * q + j] = p
                    h[2, i * q + j] = s.block_table[p // bs] * bs + p % bs
                hc[i] = len(s)
                ht[i, : len(s.block_table)] = torch.as_tensor(s.block_table, dtype=torch.int32)
            else:
                for j in range(q):
                    h[0, i * q + j] = 0
                    h[1, i * q + j] = j
                    h[2, i * q + j] = self.scratch_block * bs + j
                hc[i] = q
                ht[i, 0] = self.scratch_block
        n = b * q
        self.input_ids[:n].copy_(h[0, :n], non_blocking=True)
        self.positions[:n].copy_(h[1, :n], non_blocking=True)
        self.slots[:n].copy_(h[2, :n], non_blocking=True)
        self.ctx[:b].copy_(hc[:b], non_blocking=True)
        self.tables[:b].copy_(ht[:b], non_blocking=True)

    @torch.inference_mode()
    def run(self, seqs):
        b = self.sizes[bisect.bisect_left(self.sizes, len(seqs))]
        self._fill(seqs, b)
        g, logits, _ = self.graphs[b]
        g.replay()
        return logits[: len(seqs) * self.q]


class SpeculativeEngine:
    def __init__(self, config: EngineConfig, draft_model: str, k: int = 4, draft_blocks: int | None = None):
        self.k = k
        dev = config.device
        use_triton = config.use_triton if config.use_triton is not None else (dev == "cuda" and torch.cuda.is_available())
        fused = None
        if use_triton:
            from nimbus.kernels import rmsnorm

            fused = (rmsnorm.rms_norm, rmsnorm.add_rms_norm)
        dtype = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[config.dtype]
        # Load the draft first so the target's memory profiler sees what is left.
        self.draft_model, self.draft_cfg, _ = load_model(
            draft_model, dev, dtype, AttentionBackend(use_triton), config.max_model_len, fused)
        max_len = config.max_model_len or self.draft_cfg.max_position
        per_seq = (max_len + config.block_size - 1) // config.block_size
        nb = draft_blocks or config.max_num_seqs * per_seq
        self.use_graphs = use_triton and not config.enforce_eager and dev == "cuda"
        self.draft_kv = KVCache(self.draft_cfg, nb + int(self.use_graphs), config.block_size, dtype, dev)
        self.draft_bm = BlockManager(nb, config.block_size)
        self.target = LLMEngine(config)
        if self.target.model_cfg.vocab_size != self.draft_cfg.vocab_size:
            raise ValueError("draft and target vocabularies differ")
        self.max_model_len = self.target.max_model_len
        self.bs = config.block_size
        self.stats = SpecStats()
        self.draft_graphs = self.verify_graphs = None
        if self.use_graphs:
            from nimbus.cuda_graph import DecodeGraphs

            self.draft_graphs = DecodeGraphs(self.draft_model, self.draft_kv.layers, config.block_size, max_len,
                                             config.max_num_seqs, scratch_block=nb,
                                             num_kv_heads=self.draft_cfg.num_kv_heads, device=dev)
            self.draft_graphs.capture()
            self.verify_graphs = VerifyGraphs(self.target.model, self.target.kv.layers, config.block_size, max_len,
                                              config.max_num_seqs, k, scratch_block=self.target.bm.allocator.num_blocks,
                                              device=dev)
            self.verify_graphs.capture()

    # ---- helpers ----

    def _draft_logits(self, shadows):
        if self.draft_graphs is not None and all(sh.num_pending == 1 for sh in shadows):
            logits = self.draft_graphs.run(shadows)
            rows = dict(zip(map(id, shadows), range(len(shadows))))
        else:
            batch = sorted(((sh, sh.num_pending) for sh in shadows), key=lambda x: x[1] == 1)
            step = prepare(batch, self.bs, self.draft_kv.layers[0][0].device)
            h = self.draft_model(step.input_ids, step.positions, self.draft_kv.layers, step.meta)
            logits = self.draft_model.logits(h[step.sample_rows])
            rows = dict(zip(map(id, step.sample_seqs), range(len(step.sample_seqs))))
        for sh in shadows:
            sh.num_computed = len(sh)
        return torch.stack([logits[rows[id(sh)]] for sh in shadows])

    def _verify_logits(self, seqs):
        q = self.k + 1
        if self.verify_graphs is not None:
            return self.verify_graphs.run(seqs).view(len(seqs), q, -1)
        step = prepare([(s, q) for s in seqs], self.bs, self.target.device)
        h = self.target.model(step.input_ids, step.positions, self.target.kv.layers, step.meta)
        return self.target.model.logits(h).view(len(seqs), q, -1)

    def _stop(self, seq: Sequence, tok: int) -> SeqStatus | None:
        p = seq.params
        if not p.ignore_eos and tok in self.target.eos_ids:
            return SeqStatus.FINISHED_STOPPED
        if tok in p.stop_token_ids:
            return SeqStatus.FINISHED_STOPPED
        if len(seq.output_ids) >= p.max_tokens or len(seq) >= self.max_model_len:
            return SeqStatus.FINISHED_LENGTH
        return None

    # ---- generation ----

    @torch.inference_mode()
    def generate(self, prompts: list[list[int]], params: SamplingParams | list[SamplingParams]) -> list[list[int]]:
        tgt = self.target
        plist = params if isinstance(params, list) else [params] * len(prompts)
        seqs, shadows = [], []
        for i, (ids, sp) in enumerate(zip(prompts, plist)):
            if len(ids) + sp.max_tokens + self.k + 1 > self.max_model_len:
                raise ValueError("prompt + max_tokens + k + 1 exceeds max_model_len")
            s = Sequence(i, ids, sp)
            sh = Sequence(i, ids, sp)
            sh.token_ids = s.token_ids  # the draft's view shares the token list
            seqs.append(s)
            shadows.append(sh)

        # Prefill: the target samples the first token; the draft caches the prompt.
        for s in seqs:
            tgt.bm.grow(s, len(s))
        step = prepare([(s, len(s)) for s in seqs], self.bs, tgt.device)
        h = tgt.model(step.input_ids, step.positions, tgt.kv.layers, step.meta)
        first = tgt.sampler(tgt.model.logits(h[step.sample_rows]), seqs)
        for s in seqs:
            s.num_computed = len(s)
        for sh in shadows:
            self.draft_bm.grow(sh, len(sh))
        step = prepare([(sh, len(sh)) for sh in shadows], self.bs, tgt.device)
        self.draft_model(step.input_ids, step.positions, self.draft_kv.layers, step.meta)
        for sh in shadows:
            sh.num_computed = len(sh)

        active = []
        for s, sh, tok in zip(seqs, shadows, first):
            s.append(tok)
            if self._stop(s, tok) is None:
                active.append((s, sh))
            else:
                tgt.bm.free(s)
                self.draft_bm.free(sh)

        k = self.k
        while active:
            cur = [s for s, _ in active]
            cur_sh = [sh for _, sh in active]
            base = {id(s): len(s) for s in cur}
            sampling = any(s.params.temperature > 0 for s in cur)
            drafts = {id(s): [] for s in cur}
            qs = {id(s): [] for s in cur}

            t0 = time.perf_counter()
            chain = (not sampling and self.draft_graphs is not None
                     and all(sh.num_pending == 1 for sh in cur_sh))
            if chain:
                # All greedy and in step: draft k tokens in one GPU-side loop.
                for sh in cur_sh:
                    self.draft_bm.grow(sh, len(sh) + k - 1)
                toks = self.draft_graphs.greedy_chain(cur_sh, k).tolist()
                for s, sh, ds in zip(cur, cur_sh, toks):
                    drafts[id(s)] = ds
                    s.token_ids.extend(ds)
                    sh.num_computed = len(sh) - 1  # d_k is proposed, not yet fed
            for _ in range(0 if chain else k):
                for sh in cur_sh:
                    self.draft_bm.grow(sh, len(sh))
                logits = self._draft_logits(cur_sh)
                if sampling:
                    probs = tgt.sampler.probs(logits, cur)
                for i, s in enumerate(cur):
                    if s.params.temperature > 0:
                        d = int(torch.multinomial(probs[i], 1, generator=tgt.sampler._generator(s)))
                        qs[id(s)].append(probs[i])
                    else:
                        d = int(logits[i].argmax())
                    drafts[id(s)].append(d)
                    s.token_ids.append(d)  # raw: drafts are provisional
            t1 = time.perf_counter()

            for s in cur:
                tgt.bm.grow(s, len(s))
            vlog = self._verify_logits(cur)  # [b, k+1, V]
            if sampling:
                vprobs = tgt.sampler.probs(vlog.flatten(0, 1), [s for s in cur for _ in range(k + 1)])
                vprobs = vprobs.view(len(cur), k + 1, -1)
            argmax = vlog.argmax(-1).tolist()
            self.stats.verify_time += time.perf_counter() - t1
            self.stats.draft_time += t1 - t0

            still = []
            for i, (s, sh) in enumerate(active):
                L0 = base[id(s)]
                if s.params.temperature > 0:
                    emitted, j = accept_sampled(drafts[id(s)], torch.stack(qs[id(s)]), vprobs[i],
                                                tgt.sampler._generator(s))
                else:
                    emitted, j = accept_greedy(drafts[id(s)], argmax[i])
                del s.token_ids[L0:]  # drop provisional drafts (shared with sh)
                s.num_computed = L0 + j
                sh.num_computed = min(sh.num_computed, L0 + j)
                self.stats.rounds += 1
                self.stats.proposed += k
                self.stats.accepted += j
                done = None
                for tok in emitted:
                    s.append(tok)
                    self.stats.emitted += 1
                    done = self._stop(s, tok)
                    if done is not None:
                        break
                if done is not None:
                    s.status = done
                    tgt.bm.free(s)
                    self.draft_bm.free(sh)
                else:
                    tgt.bm.truncate(s, len(s))
                    self.draft_bm.truncate(sh, len(sh))
                    still.append((s, sh))
            active = still
        return [s.output_ids for s in seqs]
