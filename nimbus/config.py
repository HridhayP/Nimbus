"""Model and engine configuration."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RopeScaling:
    """Llama 3.1-style frequency scaling ("rope_type": "llama3")."""

    factor: float
    low_freq_factor: float
    high_freq_factor: float
    original_max_position: int


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position: int
    tie_word_embeddings: bool = False
    rope_scaling: RopeScaling | None = None
    bos_token_id: int | None = None
    eos_token_ids: tuple[int, ...] = ()

    @classmethod
    def from_hf(cls, d: dict) -> "ModelConfig":
        """Builds a config from a HuggingFace Llama config.json dict.

        transformers 5 moved RoPE settings into "rope_parameters"; older
        files keep "rope_theta" and "rope_scaling" at the top level.
        """
        rope = d.get("rope_parameters") or {}
        theta = d.get("rope_theta", rope.get("rope_theta", 10000.0))
        scaling_d = d.get("rope_scaling") or (rope if rope.get("rope_type") == "llama3" else None)
        scaling = None
        if scaling_d and scaling_d.get("rope_type", scaling_d.get("type")) == "llama3":
            scaling = RopeScaling(
                factor=scaling_d["factor"],
                low_freq_factor=scaling_d["low_freq_factor"],
                high_freq_factor=scaling_d["high_freq_factor"],
                original_max_position=scaling_d["original_max_position_embeddings"],
            )
        heads = d["num_attention_heads"]
        eos = d.get("eos_token_id")
        eos_ids = tuple(eos) if isinstance(eos, list) else ((eos,) if eos is not None else ())
        return cls(
            vocab_size=d["vocab_size"],
            hidden_size=d["hidden_size"],
            intermediate_size=d["intermediate_size"],
            num_layers=d["num_hidden_layers"],
            num_heads=heads,
            num_kv_heads=d.get("num_key_value_heads", heads),
            head_dim=d.get("head_dim") or d["hidden_size"] // heads,
            rms_norm_eps=d.get("rms_norm_eps", 1e-6),
            rope_theta=float(theta),
            max_position=d.get("max_position_embeddings", 2048),
            tie_word_embeddings=d.get("tie_word_embeddings", False),
            rope_scaling=scaling,
            bos_token_id=d.get("bos_token_id"),
            eos_token_ids=eos_ids,
        )

    @classmethod
    def from_dir(cls, path: str) -> "ModelConfig":
        with open(os.path.join(path, "config.json")) as f:
            return cls.from_hf(json.load(f))

    def kv_bytes_per_token(self, dtype_bytes: int = 2) -> int:
        return self.num_layers * 2 * self.num_kv_heads * self.head_dim * dtype_bytes


@dataclass
class EngineConfig:
    model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    dtype: str = "float16"
    device: str = "cuda"
    block_size: int = 16
    # Fraction of *currently free* GPU memory the engine may use for the KV
    # cache after weights and a profiling pass are accounted for.
    gpu_memory_utilization: float = 0.90
    num_blocks: int | None = None  # override the profiled block count
    max_num_seqs: int = 64  # running sequences per step
    max_num_batched_tokens: int = 2048  # tokens per forward pass (chunked prefill)
    max_model_len: int | None = None
    quantization: str | None = None  # None or "int8"
    use_triton: bool | None = None  # None: use Triton kernels when CUDA is available
    enforce_eager: bool = False  # disable CUDA graphs for decode steps
    seed: int = 0
    extra: dict = field(default_factory=dict)
