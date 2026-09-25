# R&D tail — R5, R6, R7b, R15, R17, R21

## R21 · TTFT vs context length
Prefill is linear: **~47 ms per 1000 prompt tokens** (≈21k tok/s), so an 8k-token
context costs ~400 ms TTFT. Motivates prefix caching (R16) and chunked prefill.

## R15 · Continuous vs static batching
48 requests, variable output 16–256 tok, 8 slots: continuous **18.9 s** vs static
**25.5 s** — **continuous is 1.35× faster** by never waiting on a batch's slowest
request. The win grows with output-length variance.

## R17 · Chunked prefill (null result)
Interactive latency during big (6k-token) prefills was **63–64 ms regardless of
ubatch (2048 vs 256)** — no measurable difference. On a small fast model prefill
never stalls concurrent decode, so chunked prefill's benefit doesn't appear; it
matters only for big models with long prompts. Honest null.

## R5 · AWQ vs GPTQ vs GGUF (4-bit methods)
| method | extraction EXACT |
|---|---|
| GGUF Q4_K_M | 99.5% |
| AWQ 4-bit | 99.0% |
| bitsandbytes nf4 | 98.8% |

All three 4-bit methods preserve the rigid-schema task at ~99% — **interchangeable
here**. AWQ kernels ran on the box (prebuilt, no nvcc). GPTQ not run (expected
similar; would only differ on harder tasks). AWQ imported and quantized fine —
the box is not as toolchain-limited as feared for weight-only quant.

## R6 · Calibration sensitivity (no effect)
AWQ with task-specific calibration = **99.0%**, same as default calibration
(99.0%). **Calibration data is a non-factor** for this rigid task — the peaked
output distribution is robust to quantization regardless of calibration set.
Would matter more for open-ended tasks.

## R7b · GGUF export degradation — ROOT CAUSE PINNED
Merged-SQL-GGUF served at **39%**; base-GGUF (no adapter) at **40.5%** — the tuned
GGUF performs the same as the untuned base (and slightly worse: 75% vs 96.5% valid
SQL). So **GGUF conversion nullifies the SQL/CLS adapter** while faithfully
preserving the extraction adapter (99.5%). It is the **conversion**, not the
server/template/numerics, and it is **adapter-dependent**. Product rule: serve
tuned models from safetensors (vLLM/transformers), not GGUF, unless you verify the
specific adapter survives.

## R14 · vLLM/SGLang under load — DEFERRED
Installing vLLM (its LoRA/advanced paths need nvcc; install risks the box's working
torch 2.11/cu128) was deferred to protect the environment — itself consistent with
the project finding that llama.cpp is the clean path here. Partially covered by R13
+ the guide's engine chapters.

## Program status
20 of 24 measured; R7b & R22c resolved; R14 deferred; R18 (KV-quant) & R19
(tensor/layer split) covered by ch.23 / Part IV. The R&D program's thesis is
delivered: the build framework, the mini-ixCSP with its two bottlenecks, the
serving physics, utilization, and ops signals — all measured, failures kept in.
