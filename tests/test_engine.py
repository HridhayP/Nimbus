import random

import pytest
import torch

from nimbus.config import EngineConfig
from nimbus.engine import LLMEngine
from nimbus.model.weights import load_model
from nimbus.sampler import Sampler, top_p_filter
from nimbus.sequence import SamplingParams, Sequence
from tests.helpers import TINYLLAMA, greedy_generate, make_tiny_llama


class IdTokenizer:
    """Stand-in tokenizer for random models: ids are already tokens."""

    eos_token_id = None

    def encode(self, s):
        return [int(x) for x in s.split()]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(map(str, ids))


def tiny_engine(path, **kw):
    cfg = EngineConfig(model=path, device="cpu", dtype="float32", use_triton=False, **kw)
    return LLMEngine(cfg, tokenizer=IdTokenizer())


@pytest.mark.parametrize("num_blocks,max_tokens_per_step,block_size", [
    (400, 2048, 16),  # everything fits
    (400, 7, 16),     # chunked prefill
    (34, 64, 4),      # 136 cached tokens for 8 sequences of up to 70: preemption
    (34, 5, 4),       # both
])
def test_engine_outputs_match_solo_greedy(tmp_path, num_blocks, max_tokens_per_step, block_size):
    """Chunked prefill (tiny token budgets) and preemption (too few blocks
    for all requests at once) must not change any output."""
    path = make_tiny_llama(str(tmp_path / "m"))
    rng = random.Random(2)
    prompts = [[rng.randrange(1, 256) for _ in range(rng.randrange(1, 50))] for _ in range(12)]
    model, cfg, _ = load_model(path, device="cpu", dtype=torch.float32)
    expected = [greedy_generate(model, cfg, p, 20, "cpu")[0] for p in prompts]

    eng = tiny_engine(path, num_blocks=num_blocks, max_num_batched_tokens=max_tokens_per_step,
                      max_num_seqs=8, max_model_len=128, block_size=block_size)
    outs = eng.generate(prompts, SamplingParams(max_tokens=20, ignore_eos=True))
    assert outs == expected
    assert eng.bm.num_free == num_blocks  # every block returned
    if num_blocks == 34 and max_tokens_per_step == 64:
        assert eng.stats.preemptions > 0, eng.stats  # the small cache really forced preemption


def test_stop_tokens_and_max_tokens(tmp_path):
    path = make_tiny_llama(str(tmp_path / "m"))
    eng = tiny_engine(path, num_blocks=64, max_model_len=128)
    [free_run] = eng.generate([[5, 6, 7]], SamplingParams(max_tokens=10, ignore_eos=True))
    stop = free_run[3]
    [stopped] = eng.generate([[5, 6, 7]], SamplingParams(max_tokens=10, stop_token_ids=[stop], ignore_eos=True))
    assert stopped == free_run[: free_run.index(stop) + 1]
    assert len(eng.generate([[5]], SamplingParams(max_tokens=3, ignore_eos=True))[0]) == 3


def test_abort_frees_blocks(tmp_path):
    path = make_tiny_llama(str(tmp_path / "m"))
    eng = tiny_engine(path, num_blocks=64, max_model_len=128)
    sid = eng.add_request([1, 2, 3, 4, 5], SamplingParams(max_tokens=50, ignore_eos=True))
    eng.step()
    eng.step()
    eng.abort(sid)
    assert eng.bm.num_free == 64 and not eng.has_unfinished()


def test_top_p_filter_keeps_smallest_nucleus():
    p = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    kept = top_p_filter(p, torch.tensor([0.75]))
    torch.testing.assert_close(kept, torch.tensor([[0.5, 0.3, 0.0, 0.0]]))
    kept = top_p_filter(p, torch.tensor([1.0]))
    torch.testing.assert_close(kept, p)


def test_seeded_sampling_is_reproducible_and_follows_distribution():
    torch.manual_seed(0)
    logits = torch.randn(1, 50)
    s = Sampler("cpu")
    seq = lambda i, seed: Sequence(i, [1], SamplingParams(temperature=0.8, seed=seed))  # noqa: E731
    a = [s([logits[0:1]][0], [seq(0, 7)])[0] for _ in range(1)]
    s2 = Sampler("cpu")
    b = [s2([logits[0:1]][0], [seq(0, 7)])[0] for _ in range(1)]
    assert a == b
    # empirical frequencies match softmax(logits / T)
    sq = seq(1, 123)
    draws = torch.tensor([s(logits, [sq])[0] for _ in range(20000)])
    freq = torch.bincount(draws, minlength=50).float() / len(draws)
    torch.testing.assert_close(freq, torch.softmax(logits[0] / 0.8, -1), atol=0.012, rtol=0)


@pytest.mark.cuda
@pytest.mark.model
def test_engine_batch_matches_hf_on_tinyllama():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from tests.test_model import PROMPTS

    tok = AutoTokenizer.from_pretrained(TINYLLAMA)
    prompts = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                       return_dict=True)["input_ids"] for p in PROMPTS]
    eng = LLMEngine(EngineConfig(use_triton=False, max_model_len=1024))
    outs = eng.generate(prompts, SamplingParams(max_tokens=32, ignore_eos=True))
    del eng
    torch.cuda.empty_cache()
    ref = AutoModelForCausalLM.from_pretrained(TINYLLAMA, dtype=torch.float16).cuda().eval()
    same = 0
    for ids, o in zip(prompts, outs):
        hf = ref.generate(torch.tensor([ids], device="cuda"), max_new_tokens=32, do_sample=False,
                          eos_token_id=None)[0, len(ids):].tolist()
        same += hf == o
    # Batched fp16 matmuls may break a near-tie differently from batch-1 HF.
    assert same >= len(prompts) - 1, f"{same}/{len(prompts)} identical"
