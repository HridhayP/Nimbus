"""OpenAI-compatible HTTP API: /v1/completions, /v1/chat/completions,
/v1/models, /health. Supports streaming (server-sent events), temperature,
top_p, max_tokens, stop strings and seed."""

from __future__ import annotations

import json
import time
import uuid
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from nimbus.async_engine import AsyncEngine
from nimbus.sequence import SamplingParams


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class _Common(BaseModel):
    model: str | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    stop: str | list[str] | None = None
    stream: bool = False
    seed: int | None = None
    n: int = Field(default=1, ge=1, le=1)


class CompletionRequest(_Common):
    prompt: str | list[int]


class ChatRequest(_Common):
    messages: list[Message]
    max_completion_tokens: int | None = Field(default=None, ge=1)


def create_app(engine: AsyncEngine, model_name: str, default_max_tokens: int = 256) -> FastAPI:
    app = FastAPI(title="nimbus")
    tok = engine.engine.tokenizer

    def params(req: _Common, max_tokens: int | None) -> SamplingParams:
        return SamplingParams(max_tokens=max_tokens or default_max_tokens, temperature=req.temperature,
                              top_p=req.top_p, seed=req.seed)

    def stops(req: _Common) -> list[str]:
        return [req.stop] if isinstance(req.stop, str) else list(req.stop or [])

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "nimbus"}]}

    async def run(prompt, sp, stop_list, kind: str):
        rid = ("cmpl-" if kind == "completion" else "chatcmpl-") + uuid.uuid4().hex[:24]
        created = int(time.time())
        gen = engine.generate(prompt, sp, stop_list)

        def chunk_obj(text, reason):
            if kind == "completion":
                return {"id": rid, "object": "text_completion", "created": created, "model": model_name,
                        "choices": [{"index": 0, "text": text, "finish_reason": reason}]}
            return {"id": rid, "object": "chat.completion.chunk", "created": created, "model": model_name,
                    "choices": [{"index": 0, "delta": {"content": text} if text else {}, "finish_reason": reason}]}

        async def stream():
            if kind == "chat":
                first = chunk_obj("", None)
                first["choices"][0]["delta"] = {"role": "assistant"}
                yield f"data: {json.dumps(first)}\n\n"
            async for c in gen:
                yield f"data: {json.dumps(chunk_obj(c.text, c.finish_reason if c.finished else None))}\n\n"
            yield "data: [DONE]\n\n"

        return rid, created, gen, stream

    async def collect(gen):
        text, last = "", None
        async for c in gen:
            text += c.text
            last = c
        return text, last

    def usage(last):
        return {"prompt_tokens": last.prompt_tokens, "completion_tokens": last.completion_tokens,
                "total_tokens": last.prompt_tokens + last.completion_tokens}

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        rid, created, gen, stream = await run(req.prompt, params(req, req.max_tokens), stops(req), "completion")
        if req.stream:
            return StreamingResponse(stream(), media_type="text/event-stream")
        try:
            text, last = await collect(gen)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        return JSONResponse({
            "id": rid, "object": "text_completion", "created": created, "model": model_name,
            "choices": [{"index": 0, "text": text, "finish_reason": last.finish_reason, "logprobs": None}],
            "usage": usage(last),
        })

    @app.post("/v1/chat/completions")
    async def chat(req: ChatRequest):
        ids = tok.apply_chat_template([m.model_dump() for m in req.messages], add_generation_prompt=True,
                                      return_dict=True)["input_ids"]
        sp = params(req, req.max_completion_tokens or req.max_tokens)
        rid, created, gen, stream = await run(list(ids), sp, stops(req), "chat")
        if req.stream:
            return StreamingResponse(stream(), media_type="text/event-stream")
        try:
            text, last = await collect(gen)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        return JSONResponse({
            "id": rid, "object": "chat.completion", "created": created, "model": model_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": last.finish_reason}],
            "usage": usage(last),
        })

    return app
