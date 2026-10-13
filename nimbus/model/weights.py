"""Loading HuggingFace Llama checkpoints (safetensors) into LlamaModel."""

from __future__ import annotations

import glob
import os

import torch
from safetensors import safe_open

from nimbus.config import ModelConfig
from nimbus.model.attention import AttentionBackend
from nimbus.model.llama import LlamaModel, Rotary


def resolve_model_path(name_or_path: str) -> str:
    if os.path.isdir(name_or_path):
        return name_or_path
    from huggingface_hub import snapshot_download

    path = snapshot_download(name_or_path, allow_patterns=["*.json", "*.safetensors", "*.model"])
    if not glob.glob(os.path.join(path, "*.safetensors")):
        # Older checkpoints (e.g. JackFram/llama-68m) ship only pytorch_model.bin.
        path = snapshot_download(name_or_path, allow_patterns=["*.json", "*.bin", "*.model"])
    return path


def _read_tensors(path: str, device: str) -> dict[str, torch.Tensor]:
    files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if not files:
        bins = sorted(glob.glob(os.path.join(path, "pytorch_model*.bin")))
        if not bins:
            raise FileNotFoundError(f"no .safetensors or pytorch_model*.bin files in {path}")
        out = {}
        for f in bins:
            # weights_only refuses to unpickle anything but tensors
            out.update(torch.load(f, map_location=device, weights_only=True))
        return out
    out = {}
    for f in files:
        with safe_open(f, framework="pt", device=device) as st:
            for k in st.keys():
                out[k] = st.get_tensor(k)
    return out


def hf_to_nimbus(hf: dict[str, torch.Tensor], cfg: ModelConfig, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Renames HF weights to ours and fuses q/k/v and gate/up."""
    def g(name):
        return hf[name].to(dtype)

    sd = {"embed_tokens.weight": g("model.embed_tokens.weight"), "norm.weight": g("model.norm.weight")}
    lm = "lm_head.weight" if "lm_head.weight" in hf and not cfg.tie_word_embeddings else "model.embed_tokens.weight"
    sd["lm_head.weight"] = g(lm)
    for i in range(cfg.num_layers):
        p = f"model.layers.{i}."
        a = p + "self_attn."
        sd[f"layers.{i}.self_attn.qkv_proj.weight"] = torch.cat(
            [g(a + "q_proj.weight"), g(a + "k_proj.weight"), g(a + "v_proj.weight")]
        )
        sd[f"layers.{i}.self_attn.o_proj.weight"] = g(a + "o_proj.weight")
        sd[f"layers.{i}.mlp.gate_up_proj.weight"] = torch.cat(
            [g(p + "mlp.gate_proj.weight"), g(p + "mlp.up_proj.weight")]
        )
        sd[f"layers.{i}.mlp.down_proj.weight"] = g(p + "mlp.down_proj.weight")
        sd[f"layers.{i}.input_layernorm.weight"] = g(p + "input_layernorm.weight")
        sd[f"layers.{i}.post_attention_layernorm.weight"] = g(p + "post_attention_layernorm.weight")
        for k in [a + n for n in ("q_proj.weight", "k_proj.weight", "v_proj.weight")]:
            hf.pop(k)  # free source tensors as we go
    return sd


def load_model(
    name_or_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    backend: AttentionBackend | None = None,
    max_position: int | None = None,
    fused_norm=None,
) -> tuple[LlamaModel, ModelConfig, str]:
    path = resolve_model_path(name_or_path)
    cfg = ModelConfig.from_dir(path)
    backend = backend or AttentionBackend(use_triton=False)
    with torch.device("meta"):
        model = LlamaModel(cfg, backend, max_position=1, fused_norm=fused_norm)
    sd = hf_to_nimbus(_read_tensors(path, device), cfg, dtype)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    missing = [m for m in missing if not m.startswith("rotary.")]
    if missing or unexpected:
        raise RuntimeError(f"weight mismatch: missing={missing} unexpected={unexpected}")
    model.rotary = Rotary(cfg, max_position or cfg.max_position).to(device)
    return model.eval(), cfg, path
