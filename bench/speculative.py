"""Speculative decoding at batch 1: acceptance rate and speedup over plain
(non-speculative) Nimbus decoding of the same target model.

    python bench/speculative.py --draft JackFram/llama-160m --k 4
    python bench/speculative.py --plain

Prompts: the 8 chat prompts from tests/test_model.py in TinyLlama's chat
template, 128 greedy tokens each, one request at a time. Rate is output
tokens / wall time over all prompts; median of --reps passes.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bench.common import MODEL, save, sync  # noqa: E402

OUT = 128


def chat_prompts():
    from transformers import AutoTokenizer

    from tests.test_model import PROMPTS

    tok = AutoTokenizer.from_pretrained(MODEL)
    return [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                    return_dict=True)["input_ids"] for p in PROMPTS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plain", action="store_true")
    ap.add_argument("--draft", default="JackFram/llama-160m")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    from nimbus.config import EngineConfig
    from nimbus.engine import LLMEngine
    from nimbus.sequence import SamplingParams
    from nimbus.spec.engine import SpeculativeEngine

    ps = chat_prompts()
    sp = SamplingParams(max_tokens=OUT, ignore_eos=True)
    cfg = EngineConfig(model=MODEL, max_model_len=512, max_num_seqs=4)
    eng = LLMEngine(cfg) if args.plain else SpeculativeEngine(cfg, args.draft, k=args.k)
    eng.generate(ps[:1], sp)  # warm-up

    rates, stats = [], None
    for _ in range(args.reps):
        if not args.plain:
            eng.stats.__init__()
        sync()
        t0 = time.perf_counter()
        n = sum(len(eng.generate([p], sp)[0]) for p in ps)
        sync()
        rates.append(n / (time.perf_counter() - t0))
        stats = None if args.plain else eng.stats
    name = "spec-plain" if args.plain else f"spec-{args.draft.split('/')[-1]}-k{args.k}"
    name += f"-{args.tag}" if args.tag else ""
    data = {"benchmark": "speculative", "batch": 1, "prompts": len(ps), "output_tokens_each": OUT,
            "tok_per_s_median": statistics.median(rates), "tok_per_s_all": rates, "args": vars(args)}
    if stats is not None:
        data.update(acceptance_rate=stats.acceptance_rate, tokens_per_round=stats.tokens_per_round,
                    rounds=stats.rounds, draft_s=stats.draft_time, verify_s=stats.verify_time)
    save(name, data)


if __name__ == "__main__":
    main()
