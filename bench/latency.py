"""Decode latency: time to first token (TTFT) and mean time per output
token after the first (TPOT), at a fixed batch size.

    python bench/latency.py nimbus --batch 1 [--quant int8] [--eager] [--no-triton]
    python bench/latency.py hf --batch 1

Every request has a 128-token prompt and generates 256 tokens. Nimbus is
timed per engine step; HF is timed as generate(max_new_tokens=1) for TTFT
and the full generate for the rest: TPOT = (full - ttft) / (256 - 1).
Median of --reps runs.
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402

from bench.common import MODEL, save, sync  # noqa: E402

PROMPT, OUT = 128, 256


def prompts(batch, seed=0):
    rng = random.Random(seed)
    return [[1] + [rng.randrange(100, 32000) for _ in range(PROMPT - 1)] for _ in range(batch)]


def nimbus_once(eng, ps):
    from nimbus.sequence import SamplingParams

    for p in ps:
        eng.add_request(p, SamplingParams(max_tokens=OUT, ignore_eos=True))
    sync()
    t0 = time.perf_counter()
    first = None
    while eng.has_unfinished():
        outs = eng.step()
        if first is None and outs:
            sync()
            first = time.perf_counter() - t0
    sync()
    total = time.perf_counter() - t0
    return first, (total - first) / (OUT - 1)


def hf_once(m, ps):
    ids = torch.tensor(ps, device="cuda")
    mask = torch.ones_like(ids)
    sync()
    t0 = time.perf_counter()
    m.generate(ids, attention_mask=mask, max_new_tokens=1, do_sample=False, pad_token_id=0)
    sync()
    ttft = time.perf_counter() - t0
    t0 = time.perf_counter()
    m.generate(ids, attention_mask=mask, max_new_tokens=OUT, min_new_tokens=OUT, do_sample=False, pad_token_id=0)
    sync()
    total = time.perf_counter() - t0
    return ttft, (total - ttft) / (OUT - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine", choices=["nimbus", "hf"])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--quant", choices=["int8"], default=None)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--no-triton", action="store_true")
    args = ap.parse_args()
    ps = prompts(args.batch)
    if args.engine == "nimbus":
        from nimbus.config import EngineConfig
        from nimbus.engine import LLMEngine

        eng = LLMEngine(EngineConfig(model=MODEL, max_model_len=512, max_num_seqs=max(16, args.batch),
                                     quantization=args.quant, enforce_eager=args.eager,
                                     use_triton=False if args.no_triton else None))
        nimbus_once(eng, ps)  # warm-up
        runs = [nimbus_once(eng, ps) for _ in range(args.reps)]
    else:
        from transformers import AutoModelForCausalLM

        m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, attn_implementation="sdpa").cuda().eval()
        hf_once(m, ps)
        runs = [hf_once(m, ps) for _ in range(args.reps)]
    ttft = [r[0] * 1000 for r in runs]
    tpot = [r[1] * 1000 for r in runs]
    name = f"latency-{args.engine}-b{args.batch}" + (f"-{args.quant}" if args.quant else "") + \
           ("-eager" if args.eager else "") + ("-notriton" if args.no_triton else "")
    save(name, {
        "benchmark": "latency", "engine": args.engine, "batch": args.batch, "prompt_tokens": PROMPT,
        "output_tokens": OUT, "reps": args.reps,
        "ttft_ms_median": statistics.median(ttft), "tpot_ms_median": statistics.median(tpot),
        "ttft_ms_all": ttft, "tpot_ms_all": tpot, "args": vars(args),
    })


if __name__ == "__main__":
    main()
