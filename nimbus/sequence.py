"""Requests as the engine sees them."""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field


@dataclass
class SamplingParams:
    max_tokens: int = 128
    temperature: float = 0.0  # 0 means greedy
    top_p: float = 1.0
    stop_token_ids: list[int] = field(default_factory=list)
    ignore_eos: bool = False  # benchmarks generate exactly max_tokens
    seed: int | None = None


class SeqStatus(enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED_STOPPED = "stop"
    FINISHED_LENGTH = "length"
    FINISHED_ABORTED = "abort"

    @property
    def finished(self) -> bool:
        return self.value in ("stop", "length", "abort")


class Sequence:
    """One generation request.

    ``num_computed`` counts tokens whose keys and values are in the cache.
    Each engine step feeds tokens ``[num_computed, num_computed + n)``; the
    step completes the sequence's pending work when that reaches
    ``len(self)``, and only then is a new token sampled. A fresh prompt is
    therefore a prefill (n > 1, possibly split across steps), a running
    sequence is a decode (n == 1), and a preempted sequence simply starts
    over from num_computed = 0 with its outputs so far as part of the prompt.
    """

    def __init__(self, seq_id: int, prompt_ids: list[int], params: SamplingParams, arrival: float | None = None):
        if not prompt_ids:
            raise ValueError("empty prompt")
        self.seq_id = seq_id
        self.prompt_ids = list(prompt_ids)
        self.output_ids: list[int] = []
        self.token_ids: list[int] = list(prompt_ids)  # prompt + output
        self.params = params
        self.status = SeqStatus.WAITING
        self.block_table: list[int] = []
        self.num_computed = 0
        self.arrival = time.perf_counter() if arrival is None else arrival
        self.first_token_time: float | None = None
        self.finish_time: float | None = None
        self.num_preemptions = 0

    def __len__(self) -> int:
        return len(self.token_ids)

    def __repr__(self) -> str:
        return f"Sequence({self.seq_id}, {self.status.value}, len={len(self)}, computed={self.num_computed})"

    @property
    def num_pending(self) -> int:
        return len(self) - self.num_computed

    @property
    def is_prefill(self) -> bool:
        return self.num_pending > 1

    def append(self, token: int) -> None:
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()
        self.output_ids.append(token)
        self.token_ids.append(token)
