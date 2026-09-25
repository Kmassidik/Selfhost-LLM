# R16 — Prefix-cache win

**Question:** how much does caching a shared prompt prefix speed serving?
**Skill:** prefix caching, KV reuse.

## Method
Many sequential requests share a long prefix (a verbose instruction block) and
differ only in a short suffix; 8-token generation, so latency is prefill-dominated.
Compare warm-request latency with `cache_prompt` on vs off, sweeping prefix length.
Qwen2.5-0.5B, one slot.

## Result

| shared prefix (~tokens) | cache OFF | cache ON | speedup |
|---|---|---|---|
| 117 | 70 ms | 64 ms | 1.1× |
| 469 | 79 ms | 66 ms | 1.2× |
| 1,408 | 120 ms | 74 ms | 1.6× |
| 3,520 | 209 ms | 99 ms | 2.1× |

## Finding
The prefix-cache win **scales with prefill cost**: negligible (1.1×) for short
prompts, 2.1× for a 3,500-token shared context. On the 0.5B prefill is very fast
(~16k tok/s, R22b), so even long prefixes give a bounded win; a slower/bigger model
(where prefill dominates more) would show a larger speedup.

## Design rule
Turn prefix caching on when requests share a **long** prefix — RAG (retrieved
context), a schema-heavy system prompt, few-shot examples, a long label list (the
multi-adapter MaaS case). For short prompts it is noise.

## Pass/fail
Bar: "% speedup measured." **Pass** — 1.1× (short) to 2.1× (3.5k-token prefix),
scaling with prefill cost. Artifacts: `serve/r16_sweep.py`.
