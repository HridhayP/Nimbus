"""Runs LLMEngine on a background thread for asyncio servers.

The engine is synchronous and blocks on the GPU, so exactly one thread
owns it and loops: take new requests and aborts from a thread-safe queue,
run one step, and hand each request's new text to its asyncio queue on the
event loop. Requests therefore join and leave the running batch at token
granularity, which is what continuous batching needs.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from dataclasses import dataclass, field

from nimbus.engine import LLMEngine
from nimbus.sequence import SamplingParams


@dataclass
class StreamChunk:
    text: str
    token_ids: list[int]
    finished: bool
    finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class _Request:
    prompt: str | list[int]
    params: SamplingParams
    stop: list[str]
    out: asyncio.Queue
    loop: asyncio.AbstractEventLoop
    seq_id: int | None = None
    text: str = ""
    sent: int = 0  # characters already streamed
    done: bool = False
    holdback: int = field(default=0)  # chars held back while they might begin a stop string


class AsyncEngine:
    def __init__(self, engine: LLMEngine):
        self.engine = engine
        self._inbox: queue.Queue = queue.Queue()
        self._requests: dict[int, _Request] = {}
        self._wake = threading.Event()
        self._stopped = False
        self._thread = threading.Thread(target=self._loop, name="nimbus-engine", daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._stopped = True
        self._wake.set()
        self._thread.join(timeout=10)

    async def generate(self, prompt: str | list[int], params: SamplingParams, stop: list[str] | None = None):
        """Async iterator of StreamChunks for one request."""
        loop = asyncio.get_running_loop()
        req = _Request(prompt, params, [s for s in (stop or []) if s], asyncio.Queue(), loop)
        req.holdback = max((len(s) - 1 for s in req.stop), default=0)
        self._inbox.put(("add", req))
        self._wake.set()
        try:
            while True:
                chunk = await req.out.get()
                if isinstance(chunk, Exception):
                    raise chunk
                yield chunk
                if chunk.finished:
                    return
        finally:
            if not req.done:  # client went away: free the sequence's blocks
                self._inbox.put(("abort", req))
                self._wake.set()

    # ---- engine thread ----

    def _post(self, req: _Request, item) -> None:
        req.loop.call_soon_threadsafe(req.out.put_nowait, item)

    def _loop(self) -> None:
        eng = self.engine
        while not self._stopped:
            while True:
                try:
                    op, req = self._inbox.get_nowait()
                except queue.Empty:
                    break
                if op == "add":
                    try:
                        req.seq_id = eng.add_request(req.prompt, req.params)
                        self._requests[req.seq_id] = req
                    except Exception as e:  # e.g. prompt too long
                        req.done = True
                        self._post(req, e)
                elif op == "abort" and req.seq_id is not None:
                    eng.abort(req.seq_id)
                    self._requests.pop(req.seq_id, None)
            if not eng.has_unfinished():
                self._wake.wait(timeout=0.05)
                self._wake.clear()
                continue
            try:
                outputs = eng.step()
            except Exception as e:  # surface engine failures to every waiting client
                for req in list(self._requests.values()):
                    req.done = True
                    self._post(req, e)
                self._requests.clear()
                continue
            for o in outputs:
                req = self._requests.get(o.seq_id)
                if req is None:
                    continue
                self._deliver(req, o)

    def _deliver(self, req: _Request, o) -> None:
        req.text += o.new_text
        finished, reason = o.finished, o.finish_reason
        cut = None
        for s in req.stop:
            i = req.text.find(s)
            if i != -1 and (cut is None or i < cut):
                cut = i
        if cut is not None:
            req.text = req.text[:cut]
            finished, reason = True, "stop"
            if not o.finished:
                self.engine.abort(o.seq_id)
        # Hold back a tail that could still turn into a stop string.
        safe = len(req.text) if finished else max(req.sent, len(req.text) - req.holdback)
        delta = req.text[req.sent:safe]
        req.sent = safe
        if finished:
            req.done = True
            self._requests.pop(o.seq_id, None)
        if delta or finished:
            self._post(req, StreamChunk(delta, o.new_token_ids, finished, reason,
                                        o.num_prompt_tokens, o.num_output_tokens))
