# serve/loadtest.py — serving under load

Closing the inference-infra gap: the guide measured single-stream tok/s
everywhere, but production is about many callers at once. `loadtest.py` fires a
concurrency sweep at an OpenAI-compatible endpoint, streaming each request to
measure TTFT (time to first token), and reports system throughput and latency
percentiles per level.

## Measured — Llama-3-8B Q4_K_M, llama.cpp, 8 slots (`--parallel 8`), 2 cards

prompt ~40 tokens, max_tokens 128, requests = concurrency × 4:

| conc | sys tok/s | per-stream tok/s | TTFT p50 | TTFT p99 | e2e p50 |
|---:|---:|---:|---:|---:|---:|
| 1 | 74 | 74 | 42 ms | 87 ms | 1.7 s |
| 2 | 139 | 69 | 65 ms | 66 ms | 1.8 s |
| 4 | 202 | 51 | 109 ms | 121 ms | 2.5 s |
| **8** | **242** | 30 | 194 ms | 222 ms | 4.2 s |
| 16 | 239 | 17 | 4461 ms | 4482 ms | 8.5 s |
| 32 | 235 | 9 | 13301 ms | 13350 ms | 17.5 s |

## What the curve says

- **Continuous batching works:** aggregate throughput scales 74 → 242 tok/s (3.3×)
  from concurrency 1 to 8, because the server packs concurrent requests into one
  batch and reads each weight once for all of them.
- **Per-stream speed falls** as concurrency rises (74 → 30 tok/s at 8): every
  caller shares the batch, so each individual response is slower even as the
  system does more total work.
- **Saturation at the slot count.** With 8 slots, throughput peaks at 8 and does
  not rise further — 16 and 32 give the same ~240 tok/s.
- **Past saturation, latency degrades, not throughput.** TTFT jumps 194 ms →
  4.5 s → 13.3 s at concurrency 8 → 16 → 32, because ~24 of 32 requests queue
  behind the 8 busy slots. The system still finishes ~240 tok/s of work; callers
  just wait longer to be admitted.

## The knob and the lesson

`--parallel` (slot count) trades throughput against tail latency. More slots =
more aggregate tok/s and more KV-cache memory, but a request that arrives to a
full server waits. You size the slots to your latency budget: pick the largest
concurrency whose TTFT p99 still meets your SLA (here, ~8), and shed or queue
load beyond it rather than letting TTFT climb into seconds.

## Run it

```
CUDA_VISIBLE_DEVICES=1,2 engines/llamacpp/build/bin/llama-server \
  -m models/gguf/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf \
  --host 10.0.0.20 --port 8089 -ngl 99 -c 16384 --parallel 8 -sm layer -a llama3 &
uv run python serve/loadtest.py --levels 1,2,4,8,16,32 --max-tokens 128
```

Works against any OpenAI-compatible endpoint — point it at vLLM or SGLang to
compare their batching under the same sweep.
