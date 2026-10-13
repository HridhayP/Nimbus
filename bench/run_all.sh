#!/usr/bin/env bash
# Runs every benchmark recorded in RESULTS.md. Each configuration runs in its
# own process so GPU memory never carries over. Raw JSON goes to results/.
# Inside WSL: source ~/venvs/nimbus/bin/activate && bash bench/run_all.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
run() { echo ">>> $*"; python "$@" 2>&1 | grep -vE "Loading weights|UserWarning|warnings.warn" | tail -n 4; }

# Throughput: nimbus (3 reps via seeds) vs HF static batching (batch sweep).
for seed in 0 1 2; do
  run bench/throughput.py nimbus --n 64 --seed $seed --tag s$seed
  run bench/throughput.py nimbus --n 256 --seed $seed --tag s$seed
done
for b in 8 16 32 64; do run bench/throughput.py hf --n 64 --batch $b --tag s0; done
for seed in 1 2; do run bench/throughput.py hf --n 64 --batch 64 --seed $seed --tag s$seed; done
for b in 32 64; do run bench/throughput.py hf --n 256 --batch $b --tag s0; done
# Equal lengths: the best case for static batching (no padding waste)
for seed in 0 1 2; do
  run bench/throughput.py nimbus --n 64 --uniform --seed $seed --tag s$seed
  run bench/throughput.py hf --n 64 --batch 64 --uniform --seed $seed --tag s$seed
done
# Ablations (n=64, seed 0)
run bench/throughput.py nimbus --n 64 --eager --tag s0
run bench/throughput.py nimbus --n 64 --no-triton --tag s0
run bench/throughput.py nimbus --n 64 --quant int8 --tag s0

# Latency
for b in 1 16; do
  run bench/latency.py hf --batch $b
  run bench/latency.py nimbus --batch $b
  run bench/latency.py nimbus --batch $b --quant int8
  run bench/latency.py nimbus --batch $b --eager
done

# Speculative decoding (batch 1). Plain runs are interleaved with the
# speculative ones (tags p1..p3) so each comparison sees the same GPU clocks.
run bench/speculative.py --plain --tag p1
pi=1
for d in ${NIMBUS_DRAFTS:-JackFram/llama-160m JackFram/llama-68m}; do
  for k in 2 4 6; do run bench/speculative.py --draft $d --k $k; done
  pi=$((pi + 1))
  run bench/speculative.py --plain --tag p$pi
done

# Quantization quality and memory
for m in hf fp16 int8; do run bench/quant.py $m; done

# Kernels
run bench/kernels.py
