"""Nimbus: a small LLM inference engine (paged KV cache, continuous batching,
Triton kernels, speculative decoding, INT8 weight-only quantization)."""

import os

# Attention temporaries grow a little every decode step as contexts lengthen.
# With PyTorch's default allocator that fragments the cache until reserved
# memory exceeds the GPU; under WSL/Windows the excess silently spills into
# system RAM and steps slow down ~10x (measured: reserved 7.5 GiB on a 6 GiB
# card). Expandable segments keep reserved memory flat. Must be set before
# CUDA is initialized, so it lives here.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

__version__ = "0.1.0"
