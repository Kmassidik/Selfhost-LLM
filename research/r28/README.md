# R28 — Chasing 50 tok/s on Ternary Bonsai 2 27B

**Question:** the glicc endpoint decodes ~33 tok/s single-stream on 2x RTX 3060 Ti (layer split).
Can it reach 50 — by batching users together, or by speculative decoding?
**Model:** Ternary-Bonsai-2-27B-PTQ1_0 (`qwen35` arch, 64 layers, vocab 248,320, MTP heads
stripped). Cards 1+2 only; card 0 untouched. Measured 2026-09-26.

## A. Concurrency (does batching raise throughput?)
HTTP, production config (`-c 131072 --parallel 4 --cache-reuse 256`), 256 tokens per stream:

| streams | per-stream tok/s | aggregate tok/s |
|---|---|---|
| 1 | 33.6 | 31.3 |
| 2 | 14.7, 14.9 | 28.3 |
| 4 | ~10.1 each | 39.2 |

Engine-level control, `llama-batched-bench` (no HTTP, pp 64 / tg 128), generation tok/s:

| batch | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| aggregate | 34.3 | 41.9 | 41.7 | **119.3** |

Batching is **step-shaped** on the ternary kernel: batch 2-4 buys only ~1.2x, batch 8 jumps to
3.5x. `--parallel 4` sits on the flat part, so two users each get roughly half speed.

## B. Speculative decoding (single stream, `-c 16384 --parallel 1`)
Two prompts: free prose (~250 words) and a code rewrite (copy a function, rename one variable).
Temperature 0, 3 reps each. "fresh" = rep 0; later reps repeat the identical request.

| config | prose tok/s | code-rewrite tok/s | draft accept |
|---|---|---|---|
| baseline (no spec) | 33.1-33.3 | 33.0 | - |
| `ngram-simple` | 33.0-33.1 (no drafts) | **51.3 / 52.6 / 52.9** | 50% |
| `ngram-mod` | 32.9 fresh -> 60.2 -> 207.8 repeat | 37.3 fresh -> 70.5 / 71.0 repeat | 29-100% |
| draft Qwen3.5-0.8B, n-max 3 | **13.3** | 27.7 | 20% / 82% |
| draft Qwen3.5-0.8B, n-max 8 | 11.5-14.2 | 35.3 / 49.6 / 40.3 | 4-8% / 39-60% |

`ngram-mod` keeps its n-gram pool across requests, so a repeated identical request copies the
previous answer (207.8 tok/s at 100% acceptance) — a ceiling for exact repeats, not general speed.

## Verdict
- **50 tok/s single-stream is reachable on copy-heavy output** (code edits, rewrites, quoting
  the context) with free n-gram speculation: `ngram-simple` 51-53 tok/s fresh, 1.6x, and no
  slowdown on prose (33.0 vs 33.1). No second model needed.
- **A draft model loses.** Even at 82% acceptance the 0.8B draft is slower (27.7): verifying
  ~4 tokens runs Bonsai at batch ~4 — the flat part of the kernel curve — so verification is
  barely cheaper per token, and the draft's own passes cost more than it saves. The stock
  0.8B also disagrees with the ternary, abliterated target (20% on prose).
- **Aggregate > 50 needs >= 8 concurrent streams** (kernel step at batch 8: 119 tok/s);
  2-4 users mostly split the same ~35-40.
- Correction to an earlier live reading: a 15.9 tok/s measurement was a second request sharing
  the slots (2 streams -> ~15 each), not thermal throttling.

## Run
```
python3 research/r28/bench.py conc http://127.0.0.1:8081 "$(cat glicc-api.token)" research/r28/results_conc.json
bash research/r28/spec_trials.sh   # stops :8081, trials on :8082, restores :8081 on exit
```
Draft model: `unsloth/Qwen3.5-0.8B-GGUF` -> `models/gguf/Qwen3.5-0.8B-Q8_0.gguf` (git-ignored).
Raw: `results_conc.json`, `results_spec.json`, `batched_bench.txt`.
