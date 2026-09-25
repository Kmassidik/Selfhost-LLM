#!/usr/bin/env bash
# L2 · vLLM, on ONE card, at float16.
#
# --dtype float16 is not optional. The weights on disk are bf16; llama.cpp and
# ollama read a GGUF that was converted to F16. Loading bf16 here would mean
# comparing engines AND number formats and reporting the sum as engine quality.
# See knowledge-base chapter 09.
set -e
B=/root/Desktop/selfhostllm
cd "$B/engines/vllm"

# ninja lives in the venv. vLLM shells out to it when compiling kernels at
# startup, so the venv's bin must be on PATH or startup dies with
# FileNotFoundError: 'ninja'.
export PATH="$B/engines/vllm/.venv/bin:$PATH"

# flashinfer compiles sampling kernels at startup with --compress-mode=size,
# an option that needs CUDA 12.8 or newer. This box has 12.0 from apt, so the
# build fails with "nvcc fatal: Unknown option". Turning the flashinfer sampler
# off makes vLLM use its own, which needs no compilation.
export VLLM_USE_FLASHINFER_SAMPLER=0

CUDA_VISIBLE_DEVICES=0 .venv/bin/vllm serve \
  "$B/models/hf/SmolLM2-360M-Instruct" \
  --served-model-name SmolLM2-360M-Instruct \
  --dtype float16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --host 0.0.0.0 --port 8000 \
  "$@"
