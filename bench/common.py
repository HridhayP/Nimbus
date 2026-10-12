"""Shared benchmark helpers: environment capture, workloads, output."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import time

import torch

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def env_info() -> dict:
    info = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "platform": platform.platform(),
    }
    try:
        import transformers
        import triton

        info["triton"] = triton.__version__
        info["transformers"] = transformers.__version__
    except ImportError:
        pass
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info["gpu"] = p.name
        info["gpu_mem_gib"] = round(p.total_memory / 2**30, 2)
        try:
            info["driver"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            pass
    return info


def save(name: str, data: dict) -> str:
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"{name}.json")
    data = {"env": env_info(), **data}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(json.dumps({k: v for k, v in data.items() if k != "env"}, indent=2))
    return path


def sharegpt_like(n: int, seed: int = 0, vocab: int = 32000, max_prompt: int = 512, max_output: int = 512):
    """Synthetic requests with ShareGPT-like lengths: log-normal prompt and
    output lengths (medians ~110 and ~180 tokens, long right tails), capped.
    Token ids are random non-special ids; with ignore_eos every request
    generates exactly its target length, so all engines do identical work."""
    rng = random.Random(seed)
    reqs = []
    for _ in range(n):
        p = min(max_prompt, max(4, int(rng.lognormvariate(4.7, 0.9))))
        o = min(max_output, max(4, int(rng.lognormvariate(5.2, 0.7))))
        reqs.append(([1] + [rng.randrange(100, vocab) for _ in range(p - 1)], o))
    return reqs


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
