# R9 + R22 — the multi-adapter serving bottleneck (deep)

R8 proved 50 adapters *fit* on one card. This asks the harder question — can you
*serve* them concurrently? — and follows the answer down to the physics.

## R9 · the bottleneck: memory density ≠ serving density

96 requests across 3 adapters (peft, one adapter active at a time), three schedules:

| strategy | wall | req/s | tok/s |
|---|---|---|---|
| homogeneous (all one adapter, batched) | 15.6 s | 6.2 | 276 |
| **mixed, naive (FIFO, switch per request)** | **182 s** | **0.5** | **9** |
| mixed, grouped (sort by adapter, 3 batches) | 8.8 s | 10.9 | 270 |

**Naive multi-adapter serving collapses throughput ~30×.** Because peft has one
active adapter, a mixed stream can't batch across adapters — every request runs as
a batch of 1. **Multi-adapter MaaS is a scheduler problem, not a memory problem.**
Adapter-aware batching (grouping) fully recovers it; the cost it adds is a batching
*window* (wait to collect same-adapter requests = latency). True S-LoRA/vLLM
batches across adapters in one kernel to avoid even that.

## R22 · the physics: why batch-1 is so wasteful (in-process, transformers)

Forced 64-token decode, sweep batch size:

| batch | wall (s) | tok/s | peak VRAM |
|---|---|---|---|
| 1 | 3.17 | 20 | 1.00 GB |
| 32 | 3.22 | 636 | 1.15 GB |
| 128 | 3.43 | 2391 | 1.59 GB |

**Wall time is flat from batch 1→32 (+1.6%) for 32× the work.** Decode reads the
weights once per step regardless of batch, so batching amortizes that read for
free — until compute-bound past ~batch 48. Naive R9 throws this entire win away.

## The engine gap (the guide's founding premise, quantified)

Same Qwen2.5-0.5B, single-stream decode: **transformers 20 tok/s vs llama.cpp
142 tok/s — 7×.** transformers eager `generate()` is Python-overhead-bound (a step
per token, no CUDA graphs); llama.cpp is not. Batching hides this: it fills the
idle GPU transformers leaves, which is why transformers batch-scales 20→2391.

## R22c · the anomaly, RESOLVED

HTTP-served llama.cpp batched decode saturated at ~295 tok/s (batch 8→128),
independent of prompt-sharing and request length. Native `llama-batched-bench`
(no HTTP, no Python client) on the same f16 0.5B settles it:

| batch | decode tok/s | prefill tok/s |
|---|---|---|
| 1 | 240 | 329 |
| 8 | 1,774 | 16,062 |
| 32 | 4,342 | 20,824 |
| 64 | 6,584 | 21,609 |
| 128 | **8,378** | 22,281 |

**The engine scales 35× (240 → 8,378 tok/s).** The ~295 ceiling was the
**llama-server HTTP path + the Python load client**, never the GPU or the kernels.

### Corrections this forces (measured > quoted)
- **R13** (242 tok/s Llama-3-8B "saturation") and **R10** (398 tok/s) were
  *client-limited*, not the box's true ceiling — both understated it.
- **R10's cost-per-token is pessimistic:** true throughput is higher, so real
  $/token is *lower* — the box is more cost-competitive than R10 reported.
- **Lesson:** your load generator can be the bottleneck. Always cross-check a
  served throughput number against a native benchmark. (This is the "use genai-perf
  / proper load tools" point from the skill research, learned the hard way.)

## Status
- R9: **done** — 30× multi-adapter collapse; scheduler fix.
- R22: **done** — batching frontier + 7× engine single-stream gap.
- R22c: **done (resolved)** — the HTTP ceiling was measurement, not hardware;
  native decode scales to 8,378 tok/s. R10/R13 flagged for re-measure with a
  native/async load path.

Artifacts: `train/r9_deep.py`, `train/r22_batchcurve.py`, `train/r22b_client.py`.
