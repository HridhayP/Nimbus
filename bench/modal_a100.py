"""Run the full benchmark suite on a Modal A100 with Llama-3-8B-Instruct.

    pip install modal && modal setup
    modal secret create huggingface HF_TOKEN=hf_...   # account with Llama 3 access
    modal run bench/modal_a100.py                      # results land in results-a100/

The draft for speculative decoding is Llama-3.2-1B-Instruct (same 128K
tokenizer as Llama 3). Model weights are cached in a Modal volume so
repeat runs don't download 16 GB again.
"""

from __future__ import annotations

import json
import os
import pathlib

import modal

REPO = pathlib.Path(__file__).resolve().parents[1]
TARGET = "meta-llama/Meta-Llama-3-8B-Instruct"
DRAFTS = "meta-llama/Llama-3.2-1B-Instruct"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.11.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("transformers>=4.50", "tokenizers", "safetensors", "accelerate", "datasets",
                 "sentencepiece", "protobuf", "fastapi", "uvicorn", "pydantic>=2", "numpy", "openai", "httpx")
    .add_local_dir(REPO, "/root/nimbus", ignore=[".git", "tmp", "results*", "**/__pycache__"])
)
hf_cache = modal.Volume.from_name("nimbus-hf-cache", create_if_missing=True)
app = modal.App("nimbus-a100", image=image)


@app.function(gpu="A100-80GB", timeout=4 * 3600, secrets=[modal.Secret.from_name("huggingface")],
              volumes={"/root/.cache/huggingface": hf_cache})
def run_suite() -> dict[str, dict]:
    import subprocess

    env = dict(os.environ, NIMBUS_MODEL=TARGET, NIMBUS_DRAFTS=DRAFTS, NIMBUS_RESULTS="/root/results",
               PYTHONPATH="/root/nimbus")
    subprocess.run(["bash", "bench/run_all.sh"], cwd="/root/nimbus", env=env, check=False)
    hf_cache.commit()
    out = {}
    for p in pathlib.Path("/root/results").glob("*.json"):
        out[p.name] = json.loads(p.read_text())
    return out


@app.local_entrypoint()
def main():
    results = run_suite.remote()
    dest = REPO / "results-a100"
    dest.mkdir(exist_ok=True)
    for name, data in results.items():
        (dest / name).write_text(json.dumps(data, indent=2))
    print(f"wrote {len(results)} result files to {dest}")
