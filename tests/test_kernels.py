import random

import pytest
import torch

from nimbus.kernels import reference

pytestmark = pytest.mark.cuda


def scattered_cache(num_blocks, block_size, kv, d, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    k = torch.randn(num_blocks, block_size, kv, d, device="cuda", dtype=torch.float16, generator=g)
    v = torch.randn(num_blocks, block_size, kv, d, device="cuda", dtype=torch.float16, generator=g)
    return k, v


def random_tables(ctx_lens, block_size, num_blocks, seed=0):
    rng = random.Random(seed)
    perm = rng.sample(range(num_blocks), num_blocks)
    tables, i = [], 0
    for c in ctx_lens:
        nb = (c + block_size - 1) // block_size
        tables.append(perm[i:i + nb])
        i += nb
    width = max(len(t) for t in tables)
    bt = torch.zeros(len(tables), width, dtype=torch.int32)
    for r, t in enumerate(tables):
        bt[r, :len(t)] = torch.tensor(t)
    return bt.cuda()


@pytest.mark.parametrize("heads,kv_heads", [(32, 4), (8, 8)])
@pytest.mark.parametrize("ctx_lens", [
    [1, 17, 64, 100, 255, 300],           # many sequences: one partition
    [1500],                               # one long sequence: split-context path
    [700, 33],                            # two sequences, split path with an empty partition
    list(range(1, 200, 7)),
])
def test_paged_decode_matches_reference(heads, kv_heads, ctx_lens):
    from nimbus.kernels.paged_attention import paged_decode_attention

    d, bs = 64, 16
    k, v = scattered_cache(800, bs, kv_heads, d)
    bt = random_tables(ctx_lens, bs, 800)
    ctx = torch.tensor(ctx_lens, dtype=torch.int32, device="cuda")
    q = torch.randn(len(ctx_lens), heads, d, device="cuda", dtype=torch.float16)
    scale = d ** -0.5
    got = paged_decode_attention(q, k, v, ctx, bt, max(ctx_lens), scale)
    want = reference.paged_decode_attention(q, k, v, ctx, bt, max(ctx_lens), scale)
    torch.testing.assert_close(got, want, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("heads,kv_heads", [(32, 4), (8, 8)])
@pytest.mark.parametrize("seqs", [
    [(5, 5), (64, 64), (130, 130)],          # fresh prompts (q_len == ctx), across BLOCK_M edges
    [(3, 40), (17, 17), (1, 9), (70, 200)],  # cached prefixes (chunked prefill / spec verify)
    [(1, 1)],
])
def test_paged_prefill_matches_reference(heads, kv_heads, seqs):
    from nimbus.kernels.prefill_attention import paged_prefill_attention

    d, bs = 64, 16
    k, v = scattered_cache(400, bs, kv_heads, d, seed=1)
    ctx_lens = [c for _, c in seqs]
    bt = random_tables(ctx_lens, bs, 400, seed=1)
    qstart = [0]
    for q_len, _ in seqs:
        qstart.append(qstart[-1] + q_len)
    q = torch.randn(qstart[-1], heads, d, device="cuda", dtype=torch.float16)
    args = (q, k, v, torch.tensor(qstart, dtype=torch.int32, device="cuda"),
            torch.tensor(ctx_lens, dtype=torch.int32, device="cuda"), bt, d ** -0.5, qstart, ctx_lens)
    got = paged_prefill_attention(*args)
    want = reference.paged_prefill_attention(*args)
    torch.testing.assert_close(got, want, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("rows,cols", [(1, 2048), (37, 2048), (5, 768), (3, 4096)])
def test_rmsnorm_matches_reference(rows, cols):
    from nimbus.kernels.rmsnorm import add_rms_norm, rms_norm

    x = torch.randn(rows, cols, device="cuda", dtype=torch.float16)
    r = torch.randn_like(x)
    w = torch.randn(cols, device="cuda", dtype=torch.float16)
    torch.testing.assert_close(rms_norm(x, w, 1e-5), reference.rms_norm(x, w, 1e-5), atol=4e-3, rtol=2e-3)
    got_y, got_r = add_rms_norm(x, r, w, 1e-5)
    want_y, want_r = reference.add_rms_norm(x, r, w, 1e-5)
    torch.testing.assert_close(got_r, want_r, atol=0, rtol=0)  # the add itself is exact
    torch.testing.assert_close(got_y, want_y, atol=4e-3, rtol=2e-3)


@pytest.mark.model
def test_triton_engine_matches_hf_on_tinyllama():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from nimbus.config import EngineConfig
    from nimbus.engine import LLMEngine
    from nimbus.sequence import SamplingParams
    from tests.helpers import TINYLLAMA
    from tests.test_model import PROMPTS

    tok = AutoTokenizer.from_pretrained(TINYLLAMA)
    prompts = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                       return_dict=True)["input_ids"] for p in PROMPTS]
    eng = LLMEngine(EngineConfig(use_triton=True, max_model_len=1024))
    outs = eng.generate(prompts, SamplingParams(max_tokens=32, ignore_eos=True))
    del eng
    torch.cuda.empty_cache()
    ref = AutoModelForCausalLM.from_pretrained(TINYLLAMA, dtype=torch.float16).cuda().eval()
    same = sum(
        ref.generate(torch.tensor([ids], device="cuda"), max_new_tokens=32, do_sample=False,
                     eos_token_id=None)[0, len(ids):].tolist() == o
        for ids, o in zip(prompts, outs)
    )
    assert same >= len(prompts) - 1, f"{same}/{len(prompts)} identical"


@pytest.mark.model
def test_cuda_graph_decode_matches_eager():
    """Requests of different lengths finish at different times, so decode
    batches shrink through many sizes and get padded to captured sizes."""
    import gc

    from nimbus.config import EngineConfig
    from nimbus.engine import LLMEngine
    from nimbus.sequence import SamplingParams

    rng = random.Random(5)
    prompts = [[1] + [rng.randrange(100, 30000) for _ in range(rng.randrange(3, 120))] for _ in range(20)]
    params = [SamplingParams(max_tokens=rng.randrange(1, 60), ignore_eos=True) for _ in prompts]
    outs = {}
    for eager in (False, True):
        eng = LLMEngine(EngineConfig(use_triton=True, enforce_eager=eager, max_model_len=512, max_num_seqs=32))
        assert (eng.graphs is None) == eager
        outs[eager] = eng.generate(prompts, params)
        assert eng.bm.num_free == eng.bm.allocator.num_blocks
        del eng
        gc.collect()
        torch.cuda.empty_cache()
    same = sum(a == b for a, b in zip(outs[False], outs[True]))
    assert [len(o) for o in outs[False]] == [p.max_tokens for p in params]
    assert same >= len(prompts) - 1, f"{same}/{len(prompts)} identical"
