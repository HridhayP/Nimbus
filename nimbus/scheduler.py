"""Iteration-level (continuous) batching.

Every engine step the scheduler rebuilds the batch from scratch:

1. Running sequences go first, so work in flight is never starved: each
   decode contributes one token and a sequence part-way through a chunked
   prefill contributes its next chunk.
2. If a running sequence needs a new KV block and none is free, the most
   recently admitted running sequence is preempted: its blocks are freed and
   it goes back to the front of the waiting queue, to be recomputed later
   (recomputation is cheaper than swapping to CPU for short sequences).
3. Waiting sequences are admitted in arrival order while the token budget,
   the sequence limit and free blocks allow. A prompt longer than the
   remaining token budget is admitted with a partial chunk (chunked
   prefill), so one long prompt can't stall every decode behind it.

Prefill work is placed before decodes in the batch, as the runner expects.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from nimbus.cache.block_manager import BlockManager
from nimbus.sequence import Sequence, SeqStatus


@dataclass
class SchedulerOutput:
    batch: list[tuple[Sequence, int]] = field(default_factory=list)
    preempted: list[Sequence] = field(default_factory=list)

    @property
    def num_tokens(self) -> int:
        return sum(n for _, n in self.batch)

    def __bool__(self) -> bool:
        return bool(self.batch)


class Scheduler:
    def __init__(self, block_manager: BlockManager, max_num_seqs: int, max_num_batched_tokens: int,
                 watermark_blocks: int = 0):
        self.bm = block_manager
        self.max_num_seqs = max_num_seqs
        self.max_tokens = max_num_batched_tokens
        # Free blocks kept back when admitting new prompts, so running
        # sequences can grow for a while before anyone is preempted.
        self.watermark = watermark_blocks
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.num_preemptions = 0

    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def finish(self, seq: Sequence, status: SeqStatus) -> None:
        seq.status = status
        self.bm.free(seq)
        if seq in self.running:
            self.running.remove(seq)
        elif seq in self.waiting:
            self.waiting.remove(seq)

    def _preempt(self, seq: Sequence, out: SchedulerOutput) -> None:
        self.running.remove(seq)
        self.bm.free(seq)
        seq.num_computed = 0
        seq.status = SeqStatus.WAITING
        seq.num_preemptions += 1
        self.num_preemptions += 1
        self.waiting.appendleft(seq)
        out.preempted.append(seq)

    def schedule(self) -> SchedulerOutput:
        out = SchedulerOutput()
        budget = self.max_tokens
        scheduled: list[tuple[Sequence, int]] = []

        # 1. running sequences, oldest first
        i = 0
        while i < len(self.running) and budget > 0:
            seq = self.running[i]
            n = min(seq.num_pending, budget)
            while not self.bm.can_grow(seq, seq.num_computed + n):
                victim = self.running[-1]
                self._preempt(victim, out)
                if victim is seq:
                    break
            else:
                self.bm.grow(seq, seq.num_computed + n)
                scheduled.append((seq, n))
                budget -= n
                i += 1
                continue
            # seq itself was preempted; self.running shrank, i stays valid

        # 2. admit waiting sequences
        while self.waiting and budget > 0 and len(self.running) < self.max_num_seqs:
            seq = self.waiting[0]
            n = min(seq.num_pending, budget)
            if not self.bm.can_grow(seq, seq.num_computed + n, reserve=self.watermark):
                break
            self.waiting.popleft()
            self.bm.grow(seq, seq.num_computed + n)
            seq.status = SeqStatus.RUNNING
            self.running.append(seq)
            scheduled.append((seq, n))
            budget -= n

        # prefill work first, then decodes (a decode feeds the last token of
        # an already-started sequence)
        def is_decode(item):
            s, n = item
            return n == 1 and s.num_computed > 0 and s.num_computed + 1 == len(s)

        out.batch = [x for x in scheduled if not is_decode(x)] + [x for x in scheduled if is_decode(x)]
        return out
