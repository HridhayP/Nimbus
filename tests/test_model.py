import pytest
import torch

from nimbus.model.weights import load_model
from tests.helpers import TINYLLAMA, greedy_generate, make_tiny_llama


def hf_model(path, device, dtype):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(path, dtype=dtype, attn_implementation="sdpa").to(device).eval()


@pytest.mark.parametrize("variant", ["gqa", "mha", "llama3-rope"])
def test_tiny_random_model_matches_hf_logits(tmp_path, variant):
    kw = {}
    if variant == "mha":
        kw["kv_heads"] = 4
    if variant == "llama3-rope":
        kw["rope_scaling"] = {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                              "high_freq_factor": 4.0, "original_max_position_embeddings": 64}
    path = make_tiny_llama(str(tmp_path / variant), **kw)
    model, cfg, _ = load_model(path, device="cpu", dtype=torch.float32)
    ref = hf_model(path, "cpu", torch.float32)

    prompt = [1, 17, 99, 3, 250, 42, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]  # spans >1 block
    out, step_logits = greedy_generate(model, cfg, prompt, 24, "cpu", block_size=4)

    with torch.no_grad():
        full = ref(torch.tensor([prompt + out])).logits[0]
    for i, lg in enumerate(step_logits):
        torch.testing.assert_close(lg, full[len(prompt) - 1 + i].float(), atol=2e-4, rtol=1e-4)
    hf_out = ref.generate(torch.tensor([prompt]), max_new_tokens=24, do_sample=False)[0, len(prompt):].tolist()
    assert out == hf_out


PROMPTS = [
    "In one sentence, what is a key-value store?",
    "Write a haiku about GPUs.",
    "List three prime numbers greater than 100.",
    "Translate 'good morning' into French and Spanish.",
    "Explain what a hash map is to a ten-year-old.",
    "def fibonacci(n):",
    "The capital of Australia is",
    "Give me a short tip for writing clean Python code.",
]


@pytest.mark.cuda
@pytest.mark.model
def test_tinyllama_greedy_matches_hf_generate():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TINYLLAMA)
    model, cfg, path = load_model(TINYLLAMA, device="cuda", dtype=torch.float16)
    ref = hf_model(path, "cuda", torch.float16)
    mismatches = []
    for p in PROMPTS:
        ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                      return_dict=True)["input_ids"]
        ours, _ = greedy_generate(model, cfg, ids, 48, "cuda")
        theirs = ref.generate(torch.tensor([ids], device="cuda"), max_new_tokens=48, do_sample=False,
                              eos_token_id=None)[0, len(ids):].tolist()
        if ours != theirs:
            first = next(i for i, (a, b) in enumerate(zip(ours, theirs)) if a != b)
            mismatches.append((p, first))
    # fp16 kernels can break a near-tie differently; allow one late divergence.
    assert len(mismatches) <= 1 and all(i >= 16 for _, i in mismatches), mismatches
