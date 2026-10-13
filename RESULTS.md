# Results

Every number here was measured on the machine below by the scripts in `bench/`. Raw JSON for every run is in `results/`, including environment details per file. Where a target was missed, the real number is reported. Exploratory numbers quoted in commit messages are superseded by this file.

## Summary vs targets

| Metric | Target | Measured | Hit? |
|---|---|---|---|
| Throughput vs HF `generate()`, concurrent requests | 3.1× | **9.2×** with ShareGPT-like mixed lengths, 64 requests (1,109 vs 121 tok/s); **16×** with 256 requests (1,726 vs 108); **1.6×** when every request has the same length, HF's best case (2,212 vs 1,370) | hit on mixed lengths, miss on equal lengths |
| Per-token decode latency vs HF | −28% | **−55%** at batch 1 (38.0 → 17.0 ms), −59% at batch 16 (45.5 → 18.6 ms); with INT8 −73% / −74% | hit |
| Speculative decoding | 2.2× at 74% acceptance | **1.29× at 17% acceptance** (llama-68m draft, k=4); 1.08× at 36% with llama-160m | miss |
| INT8 perplexity loss (WikiText-2) | < 1% | **+0.05%** (7.0952 → 7.0989) | hit |
| INT8 memory | ~50% | **56%** of fp16 weight bytes (2.05 → 1.15 GiB); peak memory during evaluation 65% (2.60 → 1.70 GiB) | hit (≈) |

The 2.2× speculative target assumes an expensive target model and a well-matched draft, such as Llama-3-8B with a 1B draft. On this laptop the target is a 1.1B model that already decodes a token in 17 ms, and the only drafts that share its tokenizer are small base models that agree with its chat-style output 12–36% of the time. The A100 scripts (below) measure the regime the target describes, but they haven't been run here.

## Hardware and software

- NVIDIA GeForce RTX 4050 Laptop GPU, 6 GB (about 4.8 GB free next to the Windows desktop), driver 610.88
- Intel Core Ultra 9 185H, 32 GB RAM, on AC power
- WSL2 Ubuntu 22.04 on Windows 11 (kernel 6.18 WSL2), Python 3.11.15
- torch 2.11.0+cu128, triton 3.6.0, transformers 5.19.0
- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` in fp16. Drafts: `JackFram/llama-160m`, `JackFram/llama-68m`.
- Measured 2026-10-12

Commands (inside WSL, from the repo root):

```
source ~/venvs/nimbus/bin/activate
bash bench/run_all.sh              # everything below; each config runs in its own process
python bench/kernels.py            # or any single script; see each script's docstring
```

## 1. Throughput

`bench/throughput.py`. All requests arrive at once, and the time is how long until all of them finish. Every engine gets identical requests and generates exactly each request's target length (`ignore_eos`), and only requested tokens count as output.

- **Nimbus:** continuous batching, up to 64 running sequences, Triton kernels, decode CUDA graphs.
- **HF:** `generate()` with SDPA attention, fp16, static batches in arrival order with left padding. A batch runs until its longest request finishes. The batch size was swept, and the best one is used.

### ShareGPT-like lengths

Prompt and output lengths are log-normal (medians about 110 and 180 tokens, capped at 512), random token ids, fixed seeds 0–2.

| Requests | Engine | Output tok/s (median) | Runs, seeds 0 / 1 / 2 | Wall time (s), seeds 0 / 1 / 2 |
|---|---|---|---|---|
| 64 | **Nimbus** | **1,109** | 1,005 / 1,109 / 1,185 | 11.8 / 11.4 / 11.8 |
| 64 | HF, batch 64 (best) | 121 | 122 / 101 / 121 | 107.6 / 113.7 / 115.5 |
| 64 | HF, batch 32 | 110 | seed 0 | 119.3 |
| 64 | HF, batch 16 | 100 | seed 0 | 131.0 |
| 64 | HF, batch 8 | 87 | seed 0 | 150.7 |
| 256 | **Nimbus** | **1,726** | 1,726 / 1,664 / 1,746 | 31.8 / 30.0 / 32.6 |
| 256 | HF, batch 64 | 108 | seed 0 | 510.0 |
| 256 | HF, batch 32 | 107 | seed 0 | 514.3 |

**Ratio: 9.2× at 64 requests (1,109 / 121), 16× at 256 (1,726 / 108).** Nimbus also ran 256 requests without a single preemption. The gap comes from what static batching wastes on mixed lengths. Every batch pads its prompts to the longest one and keeps generating until its longest output is done (up to 512 tokens), so short requests hold their slots while doing nothing. HF's per-step cost also grows with the padded batch's full context length.

### Equal lengths (HF's best case)

`--uniform`: every request has a 100-token prompt and produces 128 tokens, so static batching wastes nothing on padding or stragglers.

| Engine | Output tok/s (median) | Runs, seeds 0 / 1 / 2 |
|---|---|---|
| **Nimbus** | **2,212** | 2,213 / 2,156 / 2,226 |
| HF, batch 64 | 1,370 | 1,370 / 1,583 / 1,256 |

**Ratio: 1.6×.** This isolates per-step efficiency (paged Triton attention, CUDA graphs, fused norms) from scheduling.

### Ablations (64 requests, ShareGPT-like, seed 0)

| Configuration | Output tok/s | KV blocks |
|---|---|---|
| Full (Triton + CUDA graphs) | 1,109 | 6,224 |
| No CUDA graphs (`--eager`) | 542 | 6,224 |
| No Triton (PyTorch reference attention) | 130 | 1,175 |
| INT8 weights | 1,445 | 8,395 |

CUDA graphs double throughput because decode was CPU-bound: profiling a 64-sequence step showed about 32 ms of Python kernel launches against 15 ms of GPU work. The PyTorch reference attention copies every sequence's keys and values into temporary buffers each step, which both slows it down and shrinks the memory left for the KV cache (1,175 blocks). INT8 frees 0.9 GiB, which becomes 35% more KV blocks.

## 2. Latency

`bench/latency.py`. 128-token prompts, 256 generated tokens, median of 5 runs after a warm-up.
- **TTFT:** time to first token.
- **TPOT:** mean time per output token after the first.

| Batch | Engine | TTFT (ms) | TPOT (ms) | TPOT runs |
|---|---|---|---|---|
| 1 | HF | 60.3 | 38.01 | 51.75, 49.84, 37.20, 37.37, 38.01 |
| 1 | **Nimbus** | 41.3 | **16.96** | 16.95, 16.96, 16.97, 16.96, 16.97 |
| 1 | Nimbus INT8 | 41.9 | **10.23** | 10.23, 10.18, 10.21, 10.40, 10.32 |
| 1 | Nimbus eager (no graphs) | 37.6 | 37.75 | 42.20, 37.75, 37.39, 40.57, 37.40 |
| 16 | HF | 319.8 | 45.48 | 45.48, 47.94, 42.42, 41.67, 47.63 |
| 16 | **Nimbus** | 286.0 | **18.58** | 18.80, 18.23, 18.26, 18.76, 18.58 |
| 16 | Nimbus INT8 | 302.3 | **11.60** | 11.60, 11.58, 11.75, 11.76, 11.52 |
| 16 | Nimbus eager | 285.3 | 27.68 | 27.29, 32.05, 27.29, 27.68, 31.18 |

**Decode latency cut vs HF: 55% at batch 1, 59% at batch 16; with INT8, 73% and 74%.** Without CUDA graphs Nimbus is no faster than HF at batch 1: a single sequence is pure launch overhead. INT8 helps most here because batch-1 decode is limited by reading the weights. At 10.2 ms/token, Nimbus reads 1.15 GiB of weights per token at about 120 GB/s.

## 3. Speculative decoding

`bench/speculative.py`. Batch 1, 8 chat prompts (TinyLlama chat template), 128 greedy tokens each, median of 3 passes. Plain (non-speculative) Nimbus was measured before, between and after the speculative runs in the same session, because GPU clocks drift between sessions. Plain runs were 59.6, 58.5 and 58.4 tok/s.

| Draft | k | tok/s | vs adjacent plain | Acceptance | Tokens per round |
|---|---|---|---|---|---|
| llama-160m | 2 | 64.2 | 1.08× | 35.8% | 1.71 |
| llama-160m | 4 | 61.3 | 1.03× | 22.6% | 1.88 |
| llama-160m | 6 | 53.9 | 0.90× | 16.2% | 1.94 |
| llama-68m | 2 | 74.7 | 1.28× | 27.9% | 1.55 |
| **llama-68m** | **4** | **75.2** | **1.29×** | 16.9% | 1.66 |
| llama-68m | 6 | 70.1 | 1.20× | 11.6% | 1.68 |

Speedup is computed against the adjacent plain run: p1 = 59.6 for the 160m rows, p2 = 58.5 for the 68m rows.

The 2-layer llama-68m wins despite lower acceptance: each draft step is so cheap that even 1.6 tokens per round beats one token per target pass. Higher k proposes more but accepts proportionally less, because the drafts are base models and TinyLlama-Chat writes in a chat style. Output equality with plain greedy decoding is covered by `test_speculative_greedy_equals_target_greedy_cpu` and `test_speculative_tinyllama_matches_plain_greedy`.

## 4. INT8 weight-only quantization

`bench/quant.py`. WikiText-2 raw v1 test split, 341,469 TinyLlama tokens. Sliding window of 2048 tokens with stride 512 (HuggingFace's perplexity method). `hf` runs the same evaluation through HuggingFace's model, as a check on Nimbus's evaluator.

| Model | Perplexity | Weight memory | Peak CUDA memory during eval |
|---|---|---|---|
| HF fp16 (reference) | 7.0952 | 2.05 GiB | 2.67 GiB |
| Nimbus fp16 | 7.0952 | 2.05 GiB | 2.60 GiB |
| **Nimbus INT8** | **7.0989** | **1.15 GiB** | **1.70 GiB** |

Perplexity loss: +0.052%. Weight memory: 56% of fp16; the embedding and output head stay fp16 and account for 0.24 GiB of the 1.15. Nimbus fp16 matches HF fp16 to 5 significant digits, which validates the evaluator.

## 5. Kernels

`bench/kernels.py`, `triton.testing.do_bench` medians in µs, which flush L2 between calls. TinyLlama shapes: 32 query heads, 4 KV heads, head dim 64, hidden 2048. The reference is the PyTorch implementation in `nimbus/kernels/reference.py`; for int8 it is fp16 cuBLAS (`x @ w.T`).

| Kernel | Shape | Triton µs | Reference µs | Speedup |
|---|---|---|---|---|
| paged_decode_attention | batch=1 ctx=512 | 26.2 | 278.9 | 10.64× |
| paged_decode_attention | batch=1 ctx=2048 | 39.9 | 896.1 | 22.48× |
| paged_decode_attention | batch=16 ctx=512 | 98.3 | 3611.2 | 36.74× |
| paged_decode_attention | batch=64 ctx=512 | 348.3 | 14206.6 | 40.79× |
| paged_decode_attention | batch=64 ctx=1024 | 626.7 | 28209.3 | 45.02× |
| paged_prefill_attention | seqs=1 len=512 | 118.9 | 157.9 | 1.33× |
| paged_prefill_attention | seqs=4 len=512 | 383.4 | 615.2 | 1.60× |
| paged_prefill_attention | seqs=16 len=128 | 203.4 | 1897.3 | 9.33× |
| add_rms_norm | rows=1 hidden=2048 | 10.0 | 39.7 | 3.96× |
| add_rms_norm | rows=64 hidden=2048 | 15.5 | 63.3 | 4.09× |
| add_rms_norm | rows=2048 hidden=2048 | 273.0 | 920.5 | 3.37× |
| int8_matmul vs fp16 cuBLAS | qkv_proj M=1 N=2560 K=2048 | 65.6 | 117.0 | 1.78× |
| int8_matmul vs fp16 cuBLAS | qkv_proj M=16 | 65.4 | 118.6 | 1.81× |
| int8_matmul vs fp16 cuBLAS | qkv_proj M=64 | 73.8 | 134.4 | 1.82× |
| int8_matmul vs fp16 cuBLAS | qkv_proj M=512 | 358.6 | 345.9 | 0.96× |
| int8_matmul vs fp16 cuBLAS | gate_up_proj M=1 N=11264 K=2048 | 239.7 | 391.7 | 1.63× |
| int8_matmul vs fp16 cuBLAS | gate_up_proj M=16 | 240.0 | 470.4 | 1.96× |
| int8_matmul vs fp16 cuBLAS | gate_up_proj M=64 | 264.8 | 479.8 | 1.81× |
| int8_matmul vs fp16 cuBLAS | gate_up_proj M=512 | 1540.4 | 1463.6 | 0.95× |
| int8_matmul vs fp16 cuBLAS | down_proj M=1 N=2048 K=5632 | 126.6 | 227.1 | 1.79× |
| int8_matmul vs fp16 cuBLAS | down_proj M=16 | 128.6 | 245.3 | 1.91× |
| int8_matmul vs fp16 cuBLAS | down_proj M=64 | 163.8 | 251.0 | 1.53× |
| int8_matmul vs fp16 cuBLAS | down_proj M=512 | 795.6 | 769.8 | 0.97× |

The reference decode attention gets slower with batch and context because it copies keys and values into contiguous buffers; the Triton kernel reads them where they are. The int8 matmul wins wherever the matmul is limited by reading weights (M ≤ 64) and loses slightly at prefill sizes, where it is compute-bound. These are isolated, cold-cache timings. End to end, a 64-sequence decode step went from 15.3 to 13.7 ms with INT8, a smaller gain than these numbers suggest, because consecutive layers run with a warmer cache.

## 6. A100 / Llama-3-8B (not run)

`bench/modal_a100.py` (Modal) and `bench/colab_a100.ipynb` run this whole suite with `meta-llama/Meta-Llama-3-8B-Instruct` and a `meta-llama/Llama-3.2-1B-Instruct` draft. Neither has been run: this machine has no A100, and Llama 3 is gated on HuggingFace. To run it:

```
pip install modal && modal setup
modal secret create huggingface HF_TOKEN=hf_...     # an account that accepted the Llama 3 license
modal run bench/modal_a100.py                        # results land in results-a100/
```

## Notes on method

- Laptop GPU clocks vary between sessions: plain batch-1 decoding measured anywhere from 58 to 77 tok/s across separate sessions. That is why speculative runs are compared with plain runs taken in the same session, and why throughput and latency report medians of repeated runs.
- The first full benchmark pass had two failures: llama-68m ships only `pytorch_model.bin`, which the loader didn't read yet, and WikiText-2 moved to `Salesforce/wikitext` on the HuggingFace Hub. Both were fixed and those parts rerun. The speculative section above comes entirely from the later interleaved session.
- `ignore_eos` makes every request generate exactly its target length, so all engines do identical work. Real traffic has variable lengths, so the mixed-length workload is the realistic comparison; the equal-length workload is the lower bound, where static batching wastes nothing.
