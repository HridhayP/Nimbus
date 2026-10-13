# Interview notes

Ten likely questions about Nimbus, answered from the code. Measured numbers are in RESULTS.md.

### 1. What problem does a paged KV cache solve?

Generating each token needs the keys and values of every earlier token (the KV cache). HuggingFace `generate()` keeps each request's cache as one contiguous tensor and pads every request in a batch to the same length, so memory goes to tokens that will never exist. Nimbus allocates the cache in 16-token blocks (`cache/block_manager.py`). A request owns a block table mapping its logical blocks to physical ones and gets one more block every 16 tokens, so at most 15 slots per request are wasted. The attention kernels read keys and values through the block table (`kernels/paged_attention.py`), so a request's blocks don't need to be next to each other. With less waste, more requests fit on the GPU at once, and that's where the throughput comes from.

### 2. What is continuous batching, and how does your scheduler work?

The batch is rebuilt every step instead of once per request group (`scheduler.py`). Running sequences go first: one token each for decodes, or the next chunk of a chunked prefill. Then waiting requests are admitted in arrival order while the token budget, the sequence limit and free blocks allow. A finished request leaves immediately and its blocks go straight back to the free list, so a new request can start in the same step. HF's static batching keeps running until the longest request in the batch is done, and every slot whose request finished early is wasted.

### 3. What happens when the KV cache fills up?

The scheduler preempts the most recently admitted running sequence: it frees all of that sequence's blocks, resets `num_computed` to 0, and puts it at the front of the waiting queue. When it's re-admitted, its prompt plus everything generated so far is recomputed as one prefill. Recomputation is simpler than swapping blocks to CPU and costs less for short sequences. When admitting new prompts, a small reserve of blocks is held back so running sequences can grow for a while before anyone is preempted. `test_engine_outputs_match_solo_greedy` forces preemption and checks every output still matches its solo run token for token.

### 4. Prefill vs decode: why treat them differently?

Prefill processes a whole prompt in one pass and is compute-bound. Decode processes one token per sequence and is limited by memory bandwidth, because every step reads all the weights. They use different kernels: a FlashAttention-style tiled kernel for prefill, and a kernel that gives one GPU program to each (sequence, KV head) pair for decode. Long prompts are split into chunks that fit a per-step token budget (chunked prefill), so one long prompt can't stall every decode for a whole step. The runner puts prefill tokens before decode tokens in one flat batch, so a single forward pass handles both.

### 5. Walk me through your paged decode attention kernel.

One Triton program handles one (sequence, KV head, context partition). It loads all the query heads that share that KV head (8 in TinyLlama's grouped-query attention) as one matrix, so the score computation is a real matrix multiply on tensor cores. It then walks the context in 64-token tiles, loads keys through the block table, and keeps a running max and sum (online softmax), so the full score row never exists in memory. For long contexts with few sequences, the context is split into 512-token partitions handled by separate programs, then merged ("flash-decoding"); otherwise a single program per sequence and KV head would leave most of the GPU idle. It's tested against a PyTorch reference over odd lengths, the split path and both GQA and MHA.

### 6. Why CUDA graphs, and what made them tricky?

Profiling showed decode was limited by the CPU, not the GPU. A 64-sequence step was about 15 ms of GPU work but about 32 ms of Python launching about 400 kernels. A CUDA graph records those launches once and replays them in one call. Graphs need fixed shapes and fixed addresses, so:
- inputs live in static buffers, and each batch is padded up to a captured size;
- padding rows write into a scratch KV block that no request owns;
- the decode kernel's grid is fixed per batch size.

The bug I hit: the verify graph's tile table was a temporary that Python freed after capture. Replay then read reused memory and crashed with an illegal address. The fix was to keep each graph's metadata alive with the graph.

### 7. How does speculative decoding guarantee the same output?

A small draft model proposes k tokens, and the target scores all of them in one pass: k+1 queries over the cached prefix, the same shape as a chunked prefill. Greedy: accept drafts while each equals the target's argmax, then add the target's own token, so the output is exactly what the target alone would produce. Sampling: accept draft token d with probability min(1, p(d)/q(d)); on rejection, sample from normalize(max(0, p − q)). That provably keeps the target's distribution, and `test_rejection_sampling_preserves_target_distribution` checks it empirically over 20,000 trials. Rollback is free: rejected positions are marked uncomputed and overwritten next round.

### 8. Why is speculative decoding only 1.29× here when the target was 2.2×?

The speedup depends on the acceptance rate and on how cheap the draft is compared with the target. TinyLlama-Chat already decodes a token in about 17 ms on this GPU, and the only drafts sharing its tokenizer are small base models that agree with its chat-style output 12–36% of the time. The best measured setup was llama-68m, a 2-layer model, with k=4: 17% acceptance and 1.66 tokens per round, which gives 1.29× (75.2 vs 58.5 tok/s, measured in the same session). The 160M draft accepts more (36%) but costs more per step and only reaches 1.08×. Raising k proposes more tokens but accepts a smaller share of them, so it stops helping. Speculation pays off when the target is expensive and the draft is well aligned, such as Llama-3-8B with Llama-3.2-1B. That's what `bench/modal_a100.py` is set up to measure.

### 9. How does your INT8 quantization work, and why weight-only?

Each output row of a weight matrix gets a scale of max|w|/127 and is stored as int8. Activations stay fp16, so there's no calibration data and no activation outliers to manage. Because the scale is per output channel, x·(w_q·s)ᵀ = (x·w_qᵀ)·s, so the Triton kernel loads int8 tiles (half the bytes), converts them in registers, multiplies on tensor cores, and applies the scale once at the end. Decode is limited by memory reads, which is why weight-only quantization helps latency. In microbenchmarks the kernel is 1.5–1.9× faster than fp16 cuBLAS at decode sizes. Embeddings and the output head stay fp16. The quality cost is measured as WikiText-2 perplexity.

### 10. What was the hardest bug?

Throughput collapsed from about 450 to 63 tokens/s once the KV cache got large, and I first assumed it was ordinary out-of-memory. Logging per-step memory showed PyTorch's caching allocator *reserving* 7.5 GiB on a 6 GiB GPU. Decode temporaries grew a little every step, the allocator kept making new segments instead of reusing old ones (fragmentation), and WSL silently paged the excess to system RAM instead of failing. The fix had three parts:
- expandable segments, which keep reserved memory flat;
- a hard per-process memory cap, so the allocator frees cached blocks instead of oversubscribing;
- sizing the KV cache from a profiled worst-case decode step as well as prefill.
