import pytest
import torch

from nimbus.quant.int8 import Int8Linear, dequantize_weight, quantize_model, quantize_weight, weight_bytes
from tests.helpers import TINYLLAMA, greedy_generate, make_tiny_llama


def test_quantize_roundtrip_error_is_within_half_a_step():
    torch.manual_seed(0)
    w = torch.randn(64, 300) * torch.rand(64, 1) * 3
    q, s = quantize_weight(w)
    assert q.dtype == torch.int8 and q.abs().max() <= 127
    err = (dequantize_weight(q, s) - w).abs()
    assert (err <= s[:, None] / 2 + 1e-6).all()


def test_int8_linear_reference_matches_dequantized_matmul():
    torch.manual_seed(1)
    lin = torch.nn.Linear(96, 40, bias=False)
    x = torch.randn(5, 96)
    q = Int8Linear(lin, use_triton=False)
    torch.testing.assert_close(q(x), x @ dequantize_weight(q.weight_q, q.scale).T, atol=1e-5, rtol=1e-5)


def test_quantized_tiny_model_stays_close(tmp_path):
    from nimbus.model.weights import load_model

    path = make_tiny_llama(str(tmp_path / "m"))
    model, cfg, _ = load_model(path, device="cpu", dtype=torch.float32)
    prompt = list(range(3, 40))
    _, ref = greedy_generate(model, cfg, prompt, 8, "cpu")
    before = weight_bytes(model)
    quantize_model(model, use_triton=False)
    assert weight_bytes(model) < before
    _, got = greedy_generate(model, cfg, prompt, 8, "cpu")
    cos = torch.nn.functional.cosine_similarity(torch.stack(ref), torch.stack(got), dim=-1)
    assert cos.min() > 0.99, cos


@pytest.mark.cuda
@pytest.mark.parametrize("m", [1, 7, 64, 300])
@pytest.mark.parametrize("n,k", [(3072, 2048), (2048, 5632), (11264, 2048)])
def test_int8_kernel_matches_reference(m, n, k):
    from nimbus.kernels.int8_matmul import int8_matmul

    torch.manual_seed(2)
    w = torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.02
    q, s = quantize_weight(w)
    x = torch.randn(m, k, device="cuda", dtype=torch.float16)
    want = (x.float() @ q.float().T * s.float()).half()
    torch.testing.assert_close(int8_matmul(x, q, s), want, atol=2e-2, rtol=1e-2)


@pytest.mark.cuda
@pytest.mark.model
def test_int8_tinyllama_perplexity_close_to_fp16():
    from transformers import AutoTokenizer

    from nimbus.eval import perplexity
    from nimbus.model.attention import AttentionBackend
    from nimbus.model.weights import load_model

    text = ("The history of computing is longer than the history of computing hardware and modern computing "
            "technology and includes the history of methods intended for pen and paper or for chalk and slate, "
            "with or without the aid of tables. Digital computing is intimately tied to the representation of "
            "numbers. But long before abstractions like the number arose, there were mathematical concepts to "
            "serve the purposes of civilization.") * 3
    ids = AutoTokenizer.from_pretrained(TINYLLAMA)(text).input_ids
    model, cfg, _ = load_model(TINYLLAMA, backend=AttentionBackend(True), max_position=2048)
    fp16_bytes = weight_bytes(model)
    ppl16 = perplexity(model, cfg, ids, window=512, stride=256)
    quantize_model(model, use_triton=True)
    ppl8 = perplexity(model, cfg, ids, window=512, stride=256)
    assert weight_bytes(model) < 0.62 * fp16_bytes
    assert abs(ppl8 - ppl16) / ppl16 < 0.05, (ppl16, ppl8)
