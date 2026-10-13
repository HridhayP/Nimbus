import random

import pytest
import torch

from nimbus.cache.block_manager import BlockAllocator, BlockManager, OutOfBlocks
from nimbus.cache.kv_cache import KVCache
from nimbus.model.weights import load_model
from nimbus.runner import prepare
from nimbus.sequence import SamplingParams, Sequence
from tests.helpers import greedy_generate, make_tiny_llama


def test_allocator_invariants_under_random_ops():
    rng = random.Random(0)
    a = BlockAllocator(64)
    owned: list[int] = []
    for _ in range(5000):
        op = rng.random()
        if op < 0.5 and a.num_free:
            owned.append(a.allocate())
        elif op < 0.6 and owned:
            owned.append(a.share(rng.choice(owned)))
        elif owned:
            a.release(owned.pop(rng.randrange(len(owned))))
        # every block is free xor referenced, and counts agree
        assert sorted(set(a.free)) == sorted(a.free)
        for b in range(64):
            assert (b in a.free) == (a.refcount[b] == 0)
            assert a.refcount[b] == owned.count(b)
    for b in owned:
        a.release(b)
    assert a.num_free == 64


def test_allocator_errors():
    a = BlockAllocator(1)
    b = a.allocate()
    with pytest.raises(OutOfBlocks):
        a.allocate()
    a.release(b)
    with pytest.raises(ValueError):
        a.release(b)


def test_block_manager_grow_truncate_free():
    bm = BlockManager(10, block_size=4)
    s = Sequence(0, list(range(9)), SamplingParams())
    assert bm.blocks_needed(s, 9) == 3
    bm.grow(s, 9)
    assert len(s.block_table) == 3 and bm.num_free == 7
    bm.grow(s, 12)  # still fits in 3 blocks
    assert len(s.block_table) == 3
    bm.grow(s, 13)
    assert len(s.block_table) == 4
    bm.truncate(s, 5)
    assert len(s.block_table) == 2 and bm.num_free == 8
    bm.free(s)
    assert s.block_table == [] and bm.num_free == 10


@pytest.mark.parametrize("block_size", [4, 16])
def test_interleaved_batch_matches_individual_runs(tmp_path, block_size):
    """Sequences that prefill in random chunks while others decode, on
    shuffled (non-contiguous) blocks, produce exactly their solo outputs."""
    path = make_tiny_llama(str(tmp_path / "m"))
    model, cfg, _ = load_model(path, device="cpu", dtype=torch.float32)
    rng = random.Random(1)
    prompts = [[rng.randrange(1, 256) for _ in range(rng.randrange(1, 40))] for _ in range(6)]
    n_new = 12
    expected = [greedy_generate(model, cfg, p, n_new, "cpu", block_size)[0] for p in prompts]

    cache = KVCache(cfg, 200, block_size, torch.float32, "cpu")
    bm = BlockManager(200, block_size)
    bm.allocator.free = type(bm.allocator.free)(rng.sample(range(200), 200))  # scatter blocks
    seqs = [Sequence(i, p, SamplingParams(max_tokens=n_new)) for i, p in enumerate(prompts)]
    waiting = list(seqs)
    running: list[Sequence] = []
    while any(len(s.output_ids) < n_new for s in seqs):
        # admit up to two new sequences per step
        for _ in range(2):
            if waiting:
                running.append(waiting.pop(0))
        active = [s for s in running if len(s.output_ids) < n_new]
        batch = []
        for s in sorted(active, key=lambda s: not s.is_prefill):
            n = s.num_pending if not s.is_prefill else min(s.num_pending, rng.randrange(1, 12))
            bm.grow(s, s.num_computed + n)
            batch.append((s, n))
        step = prepare(batch, block_size, "cpu")
        h = model(step.input_ids, step.positions, cache.layers, step.meta)
        logits = model.logits(h[step.sample_rows])
        for (s, n) in batch:
            s.num_computed += n
        for row, s in enumerate(step.sample_seqs):
            s.append(int(logits[row].argmax()))
    for s, exp in zip(seqs, expected):
        assert s.output_ids == exp, s.seq_id
