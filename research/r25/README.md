# R25 — Verifying Ternary Bonsai 2 27B (frontier-model claims)

Proving or breaking the claims of a 27B ternary model (1.75 bpw, "uncensored" at run
time with 0 weight edits), measured on two machines. Full writeup:
`knowledge-base/rnd-r25.html`; measured numbers in `receipts.md`.

## Verdict (short)
Hard claims TRUE — 5.95 GB PTQ1_0 at 1.75 bpw (engine-confirmed), bit-identical pack,
exactly 129 runtime hooks, 0 weights modified, 2/2 benign refusals flipped, runs on
Apple Silicon (MLX) and CUDA (GGUF). Only their self-reported benchmark scores unverified.

## Artifacts (NOT vendored here — download yourself, Apache-2.0)
- Model: `prism-ml/Ternary-Bonsai-2-27B-gguf` (PTQ1_0, 5.95 GB) and `…-mlx-2bit` (8.61 GB).
- Runtime: `Continuum-AI-Corp/OrcaBonsai-27B-Uncensored` (refusal direction + 129-site hooks).
- Kernels: PrismML llama.cpp fork (private ternary quant types 142/143) and mlx fork.

## Scripts
- `scripts/compute_params.py` — param budget + effective bits/weight from a config.
- `scripts/run_ab.sh`, `scripts/run_ab2.sh` — MLX alpha=0 vs alpha=1 behaviour A/B (Mac).
- `scripts/test8.sh` — 8 GB context-ceiling probe (CUDA, GGUF).
- `scripts/depth2.sh` — 8 GB speed-by-depth (CUDA, GGUF).
- `eval/*.txt` — benign / over-refused / neutral prompt sets (no harmful content).

## Reproduce (box, CUDA)
Build the PrismML fork (`cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86`),
download the PTQ1_0 GGUF, then run `scripts/test8.sh` / `scripts/depth2.sh`.
Mac (MLX) track needs Apple Silicon + `pip install mlx mlx-lm` and the OrcaBonsai runtime.
