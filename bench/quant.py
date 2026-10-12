"""INT8 weight-only quantization: WikiText-2 perplexity and memory.

    python bench/quant.py fp16 | int8 | hf

Perplexity is computed on the WikiText-2 (raw, v1) test split joined with
"\\n\\n", using the sliding-window method from HuggingFace's perplexity
guide (window 2048 = TinyLlama's context, stride 512). `hf` runs the same
evaluation with HuggingFace's model as a cross-check of the evaluator.
Memory is the bytes of all weight tensors plus peak CUDA allocation during
evaluation.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402

from bench.common import MODEL, save, sync  # noqa: E402

WINDOW, STRIDE = 2048, 512


def wikitext_ids():
    from datasets import load_dataset
    from transformers import AutoTokenizer

    text = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    return AutoTokenizer.from_pretrained(MODEL)(text).input_ids


@torch.inference_mode()
def hf_ppl(ids):
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
    nll, count, prev = 0.0, 0, 0
    for begin in range(0, len(ids), STRIDE):
        end = min(begin + WINDOW, len(ids))
        x = torch.tensor([ids[begin:end]], device="cuda")
        new = end - prev
        logits = m(x).logits[0].float()
        first = max(x.shape[1] - new, 1)
        nll += torch.nn.functional.cross_entropy(logits[first - 1:-1], x[0, first:], reduction="sum").item()
        count += x.shape[1] - first
        prev = end
        if end == len(ids):
            break
    wbytes = sum(p.numel() * p.element_size() for p in m.parameters())
    return math.exp(nll / count), wbytes, count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["fp16", "int8", "hf"])
    args = ap.parse_args()
    ids = wikitext_ids()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    if args.mode == "hf":
        ppl, wbytes, scored = hf_ppl(ids)
    else:
        from nimbus.eval import perplexity
        from nimbus.kernels import rmsnorm
        from nimbus.model.attention import AttentionBackend
        from nimbus.model.weights import load_model
        from nimbus.quant.int8 import quantize_model, weight_bytes

        model, cfg, _ = load_model(MODEL, backend=AttentionBackend(True), max_position=WINDOW,
                                   fused_norm=(rmsnorm.rms_norm, rmsnorm.add_rms_norm))
        if args.mode == "int8":
            quantize_model(model, use_triton=True)
        torch.cuda.reset_peak_memory_stats()
        wbytes = weight_bytes(model)
        ppl = perplexity(model, cfg, ids, window=WINDOW, stride=STRIDE)
        scored = None
    sync()
    save(f"quant-{args.mode}", {
        "benchmark": "quantization", "mode": args.mode, "dataset": "wikitext-2-raw-v1 test",
        "tokens": len(ids), "window": WINDOW, "stride": STRIDE, "perplexity": ppl,
        "weight_bytes": wbytes, "weight_gib": wbytes / 2**30,
        "peak_cuda_alloc_gib": torch.cuda.max_memory_allocated() / 2**30,
        "eval_s": time.perf_counter() - t0, "scored_tokens": scored,
    })


if __name__ == "__main__":
    main()
