import asyncio
import socket
import threading
import time

import pytest
import uvicorn

from nimbus.async_engine import AsyncEngine
from nimbus.config import EngineConfig
from nimbus.engine import LLMEngine
from nimbus.server.app import create_app
from tests.helpers import TINYLLAMA, make_tiny_llama
from tests.test_engine import IdTokenizer


class Server:
    def __init__(self, engine: LLMEngine, name: str):
        self.async_engine = AsyncEngine(engine)
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        cfg = uvicorn.Config(create_app(self.async_engine, name), host="127.0.0.1", port=self.port, log_level="error")
        self.server = uvicorn.Server(cfg)
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        while not self.server.started:
            time.sleep(0.02)
        self.url = f"http://127.0.0.1:{self.port}/v1"

    def stop(self):
        self.server.should_exit = True
        self.thread.join(timeout=10)
        self.async_engine.shutdown()


@pytest.fixture(scope="module")
def tiny_server(tmp_path_factory):
    path = make_tiny_llama(str(tmp_path_factory.mktemp("m")))
    eng = LLMEngine(EngineConfig(model=path, device="cpu", dtype="float32", use_triton=False, max_model_len=256,
                                 num_blocks=200), tokenizer=IdTokenizer())
    srv = Server(eng, "tiny")
    yield srv
    srv.stop()


def client(srv):
    from openai import OpenAI

    return OpenAI(base_url=srv.url, api_key="unused")


def test_models_and_completion(tiny_server):
    c = client(tiny_server)
    assert [m.id for m in c.models.list().data] == ["tiny"]
    r = c.completions.create(model="tiny", prompt="5 6 7 8", max_tokens=6, temperature=0)
    assert r.choices[0].finish_reason == "length"
    assert r.usage.prompt_tokens == 4 and r.usage.completion_tokens == 6
    assert len(r.choices[0].text.split()) == 6


def test_streaming_matches_non_streaming(tiny_server):
    c = client(tiny_server)
    full = c.completions.create(model="tiny", prompt="9 10 11", max_tokens=8, temperature=0).choices[0].text
    parts, reasons = [], []
    for ch in c.completions.create(model="tiny", prompt="9 10 11", max_tokens=8, temperature=0, stream=True):
        parts.append(ch.choices[0].text)
        reasons.append(ch.choices[0].finish_reason)
    assert "".join(parts) == full
    assert reasons[-1] == "length" and all(r is None for r in reasons[:-1])


def test_stop_string_truncates(tiny_server):
    c = client(tiny_server)
    full = c.completions.create(model="tiny", prompt="1 2 3", max_tokens=10, temperature=0).choices[0].text
    words = full.split()
    stop = " " + words[3]
    r = c.completions.create(model="tiny", prompt="1 2 3", max_tokens=10, temperature=0, stop=[stop])
    assert r.choices[0].finish_reason == "stop"
    assert r.choices[0].text == full[: full.index(stop)]


def test_concurrent_requests_and_disconnect_free_blocks(tiny_server):
    from openai import AsyncOpenAI

    eng = tiny_server.async_engine.engine
    c = AsyncOpenAI(base_url=tiny_server.url, api_key="unused")

    async def one(i):
        r = await c.completions.create(model="tiny", prompt=f"{i + 1} {i + 2}", max_tokens=12, temperature=0)
        return r.choices[0].text

    async def abandoned():
        stream = await c.completions.create(model="tiny", prompt="3 4 5", max_tokens=200, temperature=0, stream=True)
        async for _ in stream:
            break  # hang up after the first chunk
        await stream.close()

    async def main():
        outs = await asyncio.gather(*(one(i) for i in range(8)), abandoned())
        return outs[:8]

    outs = asyncio.run(main())
    assert all(len(o.split()) == 12 for o in outs)
    deadline = time.time() + 10
    while (eng.has_unfinished() or eng.bm.num_free != 200) and time.time() < deadline:
        time.sleep(0.05)
    assert eng.bm.num_free == 200 and not eng.has_unfinished()


def test_prompt_too_long_is_400(tiny_server):
    import openai

    with pytest.raises(openai.BadRequestError):
        client(tiny_server).completions.create(model="tiny", prompt=" ".join(["7"] * 250), max_tokens=50)


@pytest.mark.cuda
@pytest.mark.model
def test_tinyllama_chat_server():
    srv = Server(LLMEngine(EngineConfig(max_model_len=1024, max_num_seqs=16)), TINYLLAMA)
    try:
        c = client(srv)
        r = c.chat.completions.create(model=TINYLLAMA, messages=[{"role": "user", "content": "What is 2 + 2?"}],
                                      max_tokens=32, temperature=0)
        assert "4" in r.choices[0].message.content
        text = "".join(ch.choices[0].delta.content or "" for ch in c.chat.completions.create(
            model=TINYLLAMA, messages=[{"role": "user", "content": "What is 2 + 2?"}], max_tokens=32,
            temperature=0, stream=True))
        assert text == r.choices[0].message.content
    finally:
        srv.stop()
