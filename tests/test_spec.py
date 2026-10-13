import random

import pytest
import torch

from nimbus.config import EngineConfig
from nimbus.engine import LLMEngine
from nimbus.sequence import SamplingParams
from nimbus.spec.accept import accept_greedy, accept_sampled
from nimbus.spec.engine import SpeculativeEngine
from tests.helpers import TINYLLAMA, make_tiny_llama
from tests.test_engine import IdTokenizer


def test_accept_greedy():
    assert accept_greedy([5, 6, 7], [5, 6, 7, 8]) == ([5, 6, 7, 8], 3)  # all accepted + bonus
    assert accept_greedy([5, 6, 7], [5, 9, 7, 8]) == ([5, 9], 1)  # mismatch replaced by target
    assert accept_greedy([5, 6, 7], [1, 6, 7, 8]) == ([1], 0)


def test_rejection_sampling_preserves_target_distribution():
    """With context-free p_i and q_i, emitted token i must be distributed
    exactly as p_i, whatever the draft proposes."""
    torch.manual_seed(0)
    k, v, n = 3, 6, 20000
    p = torch.softmax(torch.randn(k + 1, v) * 1.5, -1)
    q = torch.softmax(torch.randn(k, v) * 1.5, -1)
    g = torch.Generator().manual_seed(1)
    counts = torch.zeros(2, v)
    totals = torch.zeros(2)
    for _ in range(n):
        drafts = [int(torch.multinomial(q[i], 1, generator=g)) for i in range(k)]
        out, _ = accept_sampled(drafts, q, p, g)
        for pos in range(min(2, len(out))):
            counts[pos, out[pos]] += 1
            totals[pos] += 1
    for pos in range(2):
        torch.testing.assert_close(counts[pos] / totals[pos], p[pos], atol=0.012, rtol=0)


def _cpu_spec(target, draft, k):
    cfg = EngineConfig(model=target, device="cpu", dtype="float32", use_triton=False, max_model_len=200, max_num_seqs=8)
    eng = SpeculativeEngine.__new__(SpeculativeEngine)
    # Build with an IdTokenizer: tiny random models ship no tokenizer.
    import nimbus.spec.engine as se

    orig = se.LLMEngine
    se.LLMEngine = lambda c: orig(c, tokenizer=IdTokenizer())
    try:
        SpeculativeEngine.__init__(eng, cfg, draft, k=k, draft_blocks=200)
    finally:
        se.LLMEngine = orig
    return eng


@pytest.mark.parametrize("k", [1, 3, 5])
def test_speculative_greedy_equals_target_greedy_cpu(tmp_path, k):
    target = make_tiny_llama(str(tmp_path / "t"), seed=0)
    other = make_tiny_llama(str(tmp_path / "d"), seed=1)
    rng = random.Random(3)
    prompts = [[rng.randrange(1, 256) for _ in range(rng.randrange(2, 30))] for _ in range(5)]
    params = [SamplingParams(max_tokens=rng.randrange(5, 40), ignore_eos=True) for _ in prompts]

    plain = LLMEngine(EngineConfig(model=target, device="cpu", dtype="float32", use_triton=False,
                                   max_model_len=200), tokenizer=IdTokenizer())
    expected = plain.generate(prompts, params)

    for draft, min_acc in ((target, 0.95), (other, 0.0)):  # identical draft, unrelated draft
        eng = _cpu_spec(target, draft, k)
        got = eng.generate(prompts, params)
        assert got == expected, (draft, k)
        assert eng.stats.acceptance_rate >= min_acc, eng.stats
        assert eng.target.bm.num_free == eng.target.bm.allocator.num_blocks
        assert eng.draft_bm.num_free == eng.draft_bm.allocator.num_blocks


@pytest.mark.cuda
@pytest.mark.model
def test_speculative_tinyllama_matches_plain_greedy():
    import gc

    from transformers import AutoTokenizer

    from tests.test_model import PROMPTS

    tok = AutoTokenizer.from_pretrained(TINYLLAMA)
    prompts = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                       return_dict=True)["input_ids"] for p in PROMPTS]
    sp = SamplingParams(max_tokens=48, ignore_eos=True)
    plain = LLMEngine(EngineConfig(max_model_len=512, max_num_seqs=8))
    expected = plain.generate(prompts, sp)
    del plain
    gc.collect()
    torch.cuda.empty_cache()
    eng = SpeculativeEngine(EngineConfig(max_model_len=512, max_num_seqs=8), "JackFram/llama-160m", k=4)
    got = eng.generate(prompts, sp)
    same = sum(a == b for a, b in zip(got, expected))
    assert same >= len(prompts) - 1, f"{same}/{len(prompts)}"
    assert 0 < eng.stats.acceptance_rate < 1
