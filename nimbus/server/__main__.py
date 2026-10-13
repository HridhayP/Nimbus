"""python -m nimbus.server --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --port 8000"""

from __future__ import annotations

import argparse
import logging

import uvicorn

from nimbus.async_engine import AsyncEngine
from nimbus.config import EngineConfig
from nimbus.engine import LLMEngine
from nimbus.server.app import create_app


def main() -> None:
    p = argparse.ArgumentParser(description="Nimbus OpenAI-compatible server")
    p.add_argument("--model", default=EngineConfig.model)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--max-num-seqs", type=int, default=64)
    p.add_argument("--quantization", choices=["int8"], default=None)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--no-triton", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    eng = LLMEngine(EngineConfig(
        model=args.model, max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
        quantization=args.quantization, enforce_eager=args.enforce_eager,
        use_triton=False if args.no_triton else None,
    ))
    app = create_app(AsyncEngine(eng), args.model)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
