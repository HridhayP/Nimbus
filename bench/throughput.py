"""Offline throughput: N requests arrive at once; how fast do they finish?

    python bench/throughput.py nimbus --n 64 [--quant int8] [--eager] [--no-triton]
    python bench/throughput.py hf --n 64 --batch 16

All engines get the same requests (bench/common.py: ShareGPT-like lengths,
fixed seed) and generate exactly each request's target length. HF's
generate() runs static batches in arrival order with left padding; a batch
runs until its longest request is done, which is the cost of not having
iteration-level scheduling. Only the requested tokens count as output.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402

from bench.common import MODEL, save, sharegpt_like, sync  # noqa: E402


def run_nimbus(reqs, args):
    from nimbus.config import EngineConfig
    from nimbus.engine import LLMEngine
    from nimbus.sequence import SamplingParams

    eng = LLMEngine(EngineConfig(model=MODEL, max_model_len=1088, max_num_seqs=args.max_num_seqs,
                                 quantization=args.quant, enforce_eager=args.eager,
                                 use_triton=False if args.no_triton else None))
    eng.generate([reqs[0][0]], SamplingParams(max_tokens=8, ignore_eos=True))  # warm-up
    eng.stats.__init__()
    sync()
    t0 = time.perf_counter()
    outs = eng.generate([p for p, _ in reqs], [SamplingParams(max_tokens=o, ignore_eos=True) for _, o in reqs])
    sync()
    el = time.perf_counter() - t0
    assert [len(x) for x in outs] == [o for _, o in reqs]
    return el, {"kv_blocks": eng.bm.allocator.num_blocks, "steps": eng.stats.steps,
                "preemptions": eng.stats.preemptions, "triton": eng.use_triton,
                "cuda_graphs": eng.graphs is not None}


def run_hf(reqs, args):
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, attn_implementation="sdpa").cuda().eval()
    pad = 0
    w = torch.tensor([reqs[0][0][:16]], device="cuda")
    m.generate(w, attention_mask=torch.ones_like(w), max_new_tokens=8, do_sample=False, pad_token_id=pad)
    sync()
    t0 = time.perf_counter()
    for i in range(0, len(reqs), args.batch):
        batch = reqs[i:i + args.batch]
        width = max(len(p) for p, _ in batch)
        ids = torch.full((len(batch), width), pad, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for r, (p, _) in enumerate(batch):
            ids[r, width - len(p):] = torch.tensor(p)
            mask[r, width - len(p):] = 1
        new = max(o for _, o in batch)
        out = m.generate(ids.cuda(), attention_mask=mask.cuda(), max_new_tokens=new, min_new_tokens=new,
                         do_sample=False, pad_token_id=pad)
        assert out.shape[1] == width + new
    sync()
    return time.perf_counter() - t0, {"batch_size": args.batch}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine", choices=["nimbus", "hf"])
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16, help="HF static batch size")
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--quant", choices=["int8"], default=None)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--no-triton", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--uniform", action="store_true",
                    help="every request: 100-token prompt, 128 output tokens (no padding waste for HF)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.uniform:
        import random

        rng = random.Random(args.seed)
        reqs = [([1] + [rng.randrange(100, 32000) for _ in range(99)], 128) for _ in range(args.n)]
    else:
        reqs = sharegpt_like(args.n, seed=args.seed)
    el, extra = run_nimbus(reqs, args) if args.engine == "nimbus" else run_hf(reqs, args)
    out_tokens = sum(o for _, o in reqs)
    in_tokens = sum(len(p) for p, _ in reqs)
    name = f"throughput-{args.engine}-n{args.n}" + (f"-b{args.batch}" if args.engine == "hf" else "")
    name += ("-uniform" if args.uniform else "") + (f"-{args.quant}" if args.quant else "") + ("-eager" if args.eager else "") + \
            ("-notriton" if args.no_triton else "") + (f"-{args.tag}" if args.tag else "")
    save(name, {
        "benchmark": "throughput", "engine": args.engine, "requests": args.n, "seed": args.seed,
        "workload": "uniform 100 in / 128 out" if args.uniform else "sharegpt-like lognormal lengths",
        "prompt_tokens": in_tokens, "output_tokens": out_tokens, "wall_s": el,
        "output_tok_per_s": out_tokens / el, "total_tok_per_s": (in_tokens + out_tokens) / el,
        "args": vars(args), **extra,
    })


if __name__ == "__main__":
    main()
