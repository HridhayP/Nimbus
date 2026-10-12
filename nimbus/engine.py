"""The engine: one call to step() is one scheduled forward pass."""

from __future__ import annotations

import itertools
import logging
import time
from dataclasses import dataclass

import torch

from nimbus.cache.block_manager import BlockManager
from nimbus.cache.kv_cache import KVCache
from nimbus.config import EngineConfig
from nimbus.model.attention import AttentionBackend
from nimbus.model.weights import load_model
from nimbus.runner import prepare
from nimbus.sampler import Sampler
from nimbus.scheduler import Scheduler
from nimbus.sequence import SamplingParams, Sequence, SeqStatus

log = logging.getLogger("nimbus")


@dataclass
class RequestOutput:
    seq_id: int
    new_token_ids: list[int]
    new_text: str
    finished: bool
    finish_reason: str | None
    num_prompt_tokens: int
    num_output_tokens: int


@dataclass
class EngineStats:
    steps: int = 0
    prefill_tokens: int = 0
    decode_tokens: int = 0
    forward_time: float = 0.0
    preemptions: int = 0


def _dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


class LLMEngine:
    def __init__(self, config: EngineConfig, tokenizer=None):
        self.config = config
        self.device = config.device
        self.dtype = _dtype(config.dtype)
        use_triton = config.use_triton
        if use_triton is None:
            use_triton = self.device == "cuda" and torch.cuda.is_available()
        self.use_triton = use_triton
        backend = AttentionBackend(use_triton=use_triton)
        fused_norm = None
        if use_triton:
            from nimbus.kernels import rmsnorm

            fused_norm = (rmsnorm.rms_norm, rmsnorm.add_rms_norm)
        torch.manual_seed(config.seed)
        if self.device == "cuda":
            # Never let PyTorch reserve more than what is physically free:
            # past that, Windows/WSL pages GPU memory to system RAM instead
            # of failing, which is far slower than freeing cached blocks.
            free, total = torch.cuda.mem_get_info()
            cap = (torch.cuda.memory_reserved() + 0.97 * free) / total
            torch.cuda.set_per_process_memory_fraction(min(1.0, cap))
        t0 = time.perf_counter()
        self.model, self.model_cfg, path = load_model(
            config.model, self.device, self.dtype, backend, config.max_model_len, fused_norm)
        if config.quantization == "int8":
            from nimbus.quant.int8 import quantize_model

            quantize_model(self.model, use_triton=use_triton)
        self.max_model_len = config.max_model_len or self.model_cfg.max_position
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(path)
        self.tokenizer = tokenizer
        eos = set(self.model_cfg.eos_token_ids)
        if getattr(tokenizer, "eos_token_id", None) is not None:
            eos.add(tokenizer.eos_token_id)
        self.eos_ids = eos

        num_blocks = config.num_blocks or self._profile_num_blocks()
        min_blocks = (self.max_model_len + config.block_size - 1) // config.block_size
        if num_blocks < min_blocks:
            raise RuntimeError(f"only {num_blocks} KV blocks fit; one max-length sequence needs {min_blocks}")
        use_graphs = use_triton and not config.enforce_eager and self.device == "cuda"
        # With graphs, one extra block (never handed out) absorbs padding rows.
        self.kv = KVCache(self.model_cfg, num_blocks + int(use_graphs), config.block_size, self.dtype, self.device)
        self.bm = BlockManager(num_blocks, config.block_size)
        self.scheduler = Scheduler(self.bm, config.max_num_seqs, config.max_num_batched_tokens,
                                   watermark_blocks=max(1, num_blocks // 100))
        self.sampler = Sampler(self.device)
        self.graphs = None
        if use_graphs:
            from nimbus.cuda_graph import DecodeGraphs

            tg = time.perf_counter()
            self.graphs = DecodeGraphs(self.model, self.kv.layers, config.block_size, self.max_model_len,
                                       config.max_num_seqs, scratch_block=num_blocks,
                                       num_kv_heads=self.model_cfg.num_kv_heads, device=self.device)
            self.graphs.capture()
            log.info("captured %d decode graphs in %.1fs", len(self.graphs.sizes), time.perf_counter() - tg)
        self.seqs: dict[int, Sequence] = {}
        # per sequence: (prefix_offset, read_offset) for incremental detokenization
        self._detok: dict[int, tuple[int, int]] = {}
        self._ids = itertools.count()
        self.stats = EngineStats()
        log.info("engine ready in %.1fs: %d KV blocks (%.2f GiB), triton=%s, quant=%s",
                 time.perf_counter() - t0, num_blocks, self.kv.num_bytes / 2**30, use_triton, config.quantization)

    # ---- memory ----

    def _profile_num_blocks(self) -> int:
        """Sizes the KV cache from what is free after a worst-case forward."""
        cfg = self.config
        bpb = KVCache.bytes_per_block(self.model_cfg, cfg.block_size, self.dtype)
        if self.device != "cuda":
            return max(64, (self.max_model_len // cfg.block_size + 1) * 4)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        # Worst-case prefill step: a full token budget of prompts, each no
        # longer than max_model_len (positions must stay within RoPE range).
        n = cfg.max_num_batched_tokens
        lens = [min(self.max_model_len, n - i) for i in range(0, n, self.max_model_len)]
        per_seq = [(l + cfg.block_size - 1) // cfg.block_size for l in lens]
        max_blocks = (self.max_model_len + cfg.block_size - 1) // cfg.block_size
        probe = KVCache(self.model_cfg, max(sum(per_seq), max_blocks), cfg.block_size, self.dtype, self.device)
        batch, nxt = [], 0
        for i, (l, nb) in enumerate(zip(lens, per_seq)):
            seq = Sequence(-1 - i, [0] * l, SamplingParams())
            seq.block_table = list(range(nxt, nxt + nb))
            nxt += nb
            batch.append((seq, l))
        step = prepare(batch, cfg.block_size, self.device)
        h = self.model(step.input_ids, step.positions, probe.layers, step.meta)
        self.model.logits(h[step.sample_rows])
        # Worst-case decode step: max_num_seqs sequences at max_model_len.
        # They all alias the same blocks, which is fine for measuring memory.
        batch = []
        for i in range(cfg.max_num_seqs):
            seq = Sequence(-1000 - i, [0] * self.max_model_len, SamplingParams())
            seq.block_table = list(range(max_blocks))
            seq.num_computed = self.max_model_len - 1
            batch.append((seq, 1))
        step = prepare(batch, cfg.block_size, self.device)
        h = self.model(step.input_ids, step.positions, probe.layers, step.meta)
        self.model.logits(h[step.sample_rows])
        torch.cuda.synchronize()
        activation_peak = torch.cuda.max_memory_allocated() - base - probe.num_bytes
        del probe, h, step
        torch.cuda.empty_cache()
        free, _ = torch.cuda.mem_get_info()
        budget = free * cfg.gpu_memory_utilization - activation_peak
        num = int(budget // bpb)
        log.info("profiled: activations %.0f MiB, free %.2f GiB -> %d blocks",
                 activation_peak / 2**20, free / 2**30, num)
        return num

    # ---- requests ----

    def add_request(self, prompt: str | list[int], params: SamplingParams | None = None,
                    arrival: float | None = None) -> int:
        params = params or SamplingParams()
        ids = self.tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
        if len(ids) + params.max_tokens > self.max_model_len:
            raise ValueError(f"prompt ({len(ids)}) + max_tokens ({params.max_tokens}) exceeds "
                             f"max_model_len {self.max_model_len}")
        seq = Sequence(next(self._ids), ids, params, arrival)
        self.seqs[seq.seq_id] = seq
        self._detok[seq.seq_id] = (0, 0)
        self.scheduler.add(seq)
        return seq.seq_id

    def abort(self, seq_id: int) -> None:
        seq = self.seqs.pop(seq_id, None)
        if seq is not None and not seq.status.finished:
            self.scheduler.finish(seq, SeqStatus.FINISHED_ABORTED)
            self.sampler.forget(seq)
            self._detok.pop(seq_id, None)

    def has_unfinished(self) -> bool:
        return self.scheduler.has_work()

    # ---- the step ----

    def step(self) -> list[RequestOutput]:
        sched = self.scheduler.schedule()
        self.stats.preemptions += len(sched.preempted)
        if not sched:
            return []
        t0 = time.perf_counter()
        if self.graphs is not None and self.graphs.can_run(sched.batch):
            sample_seqs = [s for s, _ in sched.batch]
            logits = self.graphs.run(sample_seqs)
        else:
            step = prepare(sched.batch, self.config.block_size, self.device)
            h = self.model(step.input_ids, step.positions, self.kv.layers, step.meta)
            sample_seqs = step.sample_seqs
            logits = self.model.logits(h[step.sample_rows]) if sample_seqs else None
        for seq, n in sched.batch:
            seq.num_computed += n
        tokens = self.sampler(logits, sample_seqs) if sample_seqs else []
        self.stats.forward_time += time.perf_counter() - t0
        self.stats.steps += 1
        for seq, n in sched.batch:
            if n == 1 and seq.num_computed > 1 and seq.num_computed == len(seq):
                self.stats.decode_tokens += 1
            else:
                self.stats.prefill_tokens += n
        return [self._emit(seq, tok) for seq, tok in zip(sample_seqs, tokens)]

    def _emit(self, seq: Sequence, tok: int) -> RequestOutput:
        seq.append(tok)
        p = seq.params
        reason = None
        if not p.ignore_eos and tok in self.eos_ids:
            reason = SeqStatus.FINISHED_STOPPED
        elif tok in p.stop_token_ids:
            reason = SeqStatus.FINISHED_STOPPED
        elif len(seq.output_ids) >= p.max_tokens or len(seq) >= self.max_model_len:
            reason = SeqStatus.FINISHED_LENGTH
        delta = self._detokenize(seq, final=reason is not None)
        if reason is not None:
            seq.finish_time = time.perf_counter()
            self.scheduler.finish(seq, reason)
            self.sampler.forget(seq)
            self._detok.pop(seq.seq_id, None)
            self.seqs.pop(seq.seq_id, None)
        return RequestOutput(seq.seq_id, [tok], delta, reason is not None,
                             reason.value if reason else None, len(seq.prompt_ids), len(seq.output_ids))

    def _detokenize(self, seq: Sequence, final: bool) -> str:
        """Text added by the newest token, decoding only a short window.

        Decoding the whole output every step is quadratic in its length.
        Tokenizers don't decode tokens independently (SentencePiece leading
        spaces, multi-byte characters split across tokens), so decode the
        window [prefix_offset, end) and emit what extends the decoding of
        [prefix_offset, read_offset); an incomplete character (U+FFFD) waits
        for the next token unless the sequence is finishing.
        """
        ids = seq.output_ids
        prefix_off, read_off = self._detok[seq.seq_id]
        dec = self.tokenizer.decode
        prefix = dec(ids[prefix_off:read_off], skip_special_tokens=True)
        new = dec(ids[prefix_off:], skip_special_tokens=True)
        if len(new) > len(prefix) and (final or not new.endswith("�")):
            self._detok[seq.seq_id] = (read_off, len(ids))
            return new[len(prefix):]
        return ""

    # ---- offline convenience ----

    def generate(self, prompts: list[str | list[int]], params: SamplingParams | list[SamplingParams]) -> list[list[int]]:
        """Runs prompts to completion; returns output token ids in prompt order."""
        plist = params if isinstance(params, list) else [params] * len(prompts)
        ids = [self.add_request(p, sp) for p, sp in zip(prompts, plist)]
        outs: dict[int, list[int]] = {i: [] for i in ids}
        while self.has_unfinished():
            for o in self.step():
                outs[o.seq_id].extend(o.new_token_ids)
        return [outs[i] for i in ids]
