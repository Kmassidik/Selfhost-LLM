#!/usr/bin/env bash
# L3 · SGLang, on ONE card, at float16.
#
# Same rules as the other engines: one card, 16-bit, bound to the network.
# See knowledge-base chapter 09 for why --dtype float16 is not optional.
set -e
B=/root/Desktop/selfhostllm
cd "$B/engines/sglang"

# flashinfer, SGLang's default backend, compiles kernels with --compress-mode=size,
# which needs CUDA 12.8+. This box has 12.0, so nvcc rejects it. Triton compiles
# without nvcc, and the pytorch sampler avoids the same problem in sampling.
#
# ninja is in the venv and the kernel compile runs as a subprocess
export PATH="$B/engines/sglang/.venv/bin:$PATH"

CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m sglang.launch_server \
  --model-path "$B/models/hf/SmolLM2-360M-Instruct" \
  --served-model-name SmolLM2-360M-Instruct \
  --dtype float16 \
  --context-length 8192 \
  --mem-fraction-static 0.85 \
  --attention-backend triton \
  --sampling-backend pytorch \
  --host 0.0.0.0 --port 30000 \
  "$@"
