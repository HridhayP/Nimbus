"""Kernel micro-benchmarks on TinyLlama shapes (32 query heads, 4 KV heads,
head dim 64, hidden 2048): Triton vs the PyTorch reference, and the int8
matmul vs cuBLAS fp16. Times are triton.testing.do_bench medians (us).

    python bench/kernels.py
"""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402
from triton.testing import do_bench  # noqa: E402

from bench.common import save  # noqa: E402
from nimbus.kernels import reference  # noqa: E402
from nimbus.kernels.int8_matmul import int8_matmul  # noqa: E402
from nimbus.kernels.paged_attention import paged_decode_attention  # noqa: E402
from nimbus.kernels.prefill_attention import paged_prefill_attention  # noqa: E402
from nimbus.kernels.rmsnorm import add_rms_norm  # noqa: E402
from nimbus.quant.int8 import quantize_weight  # noqa: E402

H, KV, D, BS = 32, 4, 64, 16


def us(fn):
    return do_bench(fn, warmup=25, rep=200) * 1000


def cache(blocks):
    k = torch.randn(blocks, BS, KV, D, device="cuda", dtype=torch.float16)
    return k, torch.randn_like(k)


def tables(lens, blocks):
    perm = random.Random(0).sample(range(blocks), blocks)
    width = max((c + BS - 1) // BS for c in lens)
    bt = torch.zeros(len(lens), width, dtype=torch.int32)
    i = 0
    for r, c in enumerate(lens):
        nb = (c + BS - 1) // BS
        bt[r, :nb] = torch.tensor(perm[i:i + nb])
        i += nb
    return bt.cuda()


def main():
    rows = []
    scale = D ** -0.5
    for b, ctx in [(1, 512), (1, 2048), (16, 512), (64, 512), (64, 1024)]:
        k, v = cache(b * ((ctx + BS - 1) // BS) + 8)
        bt = tables([ctx] * b, k.shape[0])
        cl = torch.full((b,), ctx, dtype=torch.int32, device="cuda")
        q = torch.randn(b, H, D, device="cuda", dtype=torch.float16)
        t = us(lambda: paged_decode_attention(q, k, v, cl, bt, ctx, scale))
        r = us(lambda: reference.paged_decode_attention(q, k, v, cl, bt, ctx, scale))
        rows.append({"kernel": "paged_decode_attention", "shape": f"batch={b} ctx={ctx}", "triton_us": t,
                     "reference_us": r, "speedup": r / t})
    for n_seq, q_len in [(1, 512), (4, 512), (16, 128)]:
        k, v = cache(n_seq * ((q_len + BS - 1) // BS) + 8)
        lens = [q_len] * n_seq
        bt = tables(lens, k.shape[0])
        qs = [i * q_len for i in range(n_seq + 1)]
        q = torch.randn(qs[-1], H, D, device="cuda", dtype=torch.float16)
        args = (q, k, v, torch.tensor(qs, dtype=torch.int32, device="cuda"),
                torch.tensor(lens, dtype=torch.int32, device="cuda"), bt, scale, qs, lens)
        t = us(lambda: paged_prefill_attention(*args))
        r = us(lambda: reference.paged_prefill_attention(*args))
        rows.append({"kernel": "paged_prefill_attention", "shape": f"seqs={n_seq} len={q_len}", "triton_us": t,
                     "reference_us": r, "speedup": r / t})
    for m in (1, 64, 2048):
        x = torch.randn(m, 2048, device="cuda", dtype=torch.float16)
        res = torch.randn_like(x)
        w = torch.randn(2048, device="cuda", dtype=torch.float16)
        t = us(lambda: add_rms_norm(x, res, w, 1e-5))
        r = us(lambda: reference.add_rms_norm(x, res, w, 1e-5))
        rows.append({"kernel": "add_rms_norm", "shape": f"rows={m} hidden=2048", "triton_us": t,
                     "reference_us": r, "speedup": r / t})
    for name, n, kdim in [("qkv_proj", 2560, 2048), ("gate_up_proj", 11264, 2048), ("down_proj", 2048, 5632)]:
        w = torch.randn(n, kdim, device="cuda", dtype=torch.float16) * 0.02
        wq, s = quantize_weight(w)
        for m in (1, 16, 64, 512):
            x = torch.randn(m, kdim, device="cuda", dtype=torch.float16)
            t = us(lambda: int8_matmul(x, wq, s))
            r = us(lambda: x @ w.T)
            rows.append({"kernel": "int8_matmul vs fp16 cuBLAS", "shape": f"{name} M={m} N={n} K={kdim}",
                         "triton_us": t, "reference_us": r, "speedup": r / t})
    for row in rows:
        print(f"{row['kernel']:28s} {row['shape']:36s} triton {row['triton_us']:9.1f}us  "
              f"ref {row['reference_us']:9.1f}us  x{row['speedup']:.2f}")
    save("kernels", {"benchmark": "kernels", "rows": rows})


if __name__ == "__main__":
    main()
