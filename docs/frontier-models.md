# Frontier models on the box (F-track)

Verifying and serving 2026 frontier models on the three-card box. Prose write-ups live in
the knowledge base on the Mac; this is the box-side summary and where the code is.

## What was done
- **Verify a shipped ternary model (R25).** Ternary Bonsai 2 27B (PTQ1_0, 5.95 GB, 1.75 bpw):
  claims check out (bit-identical pack, 129 runtime hooks, runs on Apple Silicon and CUDA).
  Code: `research/r25/`.
- **How it is built (R26/R26b).** Ternary QAT + packing: naive rounding collapses (7,000,000x),
  scaling helps but only training recovers it; packing is a free ~10x size win, speed needs a
  fused kernel. Code: `research/r26/`.
- **The kernel (R26c).** From the PrismML fork: 5 trits/byte, base-3 unpack via
  `__byte_perm`/`__vsub4`, dequant in shared memory then matmul. The fast kernel is the moat.
- **Serve it (R25c).** OpenAI-compatible, token-gated, multi-slot, prefix-cached endpoint --
  the DeepSeek/OpenRouter pattern. Scripts: `deploy/`.
- **Typed decisions (R27).** Code: `research/r27/`.
- **Chasing 50 tok/s (R28).** Batching vs speculative decoding on the endpoint. Free n-gram
  speculation hits ~52 tok/s on copy-heavy output; a 0.8B draft model is slower. Code: `research/r28/`.
- **From scratch.** A GPT, a diffusion model, an audio LM built by hand: `from-scratch/`.

## The serving endpoint (glicc-model-testing), 2 cards
```
CUDA_VISIBLE_DEVICES=1,2 engines/prismml-llamacpp/build/bin/llama-server \
  -m models/gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf -ngl 99 -sm layer -fa on \
  -ctk q4_0 -ctv q4_0 -c 131072 --parallel 4 --cache-reuse 256 \
  --api-key "$(cat glicc-api.token)" -a glicc-model-testing --host 127.0.0.1 --port 8081
```
Token in `glicc-api.token` (git-ignored). Public hostname: `deploy/expose-llm.sh` (Cloudflare
tunnel). The PrismML fork (`engines/prismml-llamacpp/`, git-ignored) is built from source --
no AVX2 on this box, so prebuilt binaries crash.

## Measured (2 cards)
- ~32 tok/s decode, 0.69 s TTFT; prefix cache: repeat 3k context TTFT 4.54s -> 1.07s (4.2x).
- Batching is step-shaped on the ternary kernel (R28): native batch 1/2/4/8 -> 34/42/42/119 tok/s;
  with `--parallel 4`, two users each get ~15 tok/s.
- Speculative (R28): `--spec-type ngram-simple` -> 51-53 tok/s on code rewrites, unchanged on prose;
  draft model Qwen3.5-0.8B -> 13-28 tok/s (slower than the 33 baseline).
- Context ceiling: **262k** full speed; **768k** via KV-in-RAM + YaRN (~2-3 tok/s);
  **1M** with ~12 layers on CPU + ubatch 8 + KV-in-RAM + threads (~1 tok/s, YaRN-degraded past 262k).

## Notes
- Secrets (`*.token`, `.env*`) and huge clones (`engines/*`) are git-ignored.
- Teaching/prose versions live in the Mac knowledge base (`knowledge-base/`), not on the box.

## Calling the API (auth)
Every request needs the bearer token -- the endpoint runs with `--api-key`, so a call with no
token returns **401**. The token value lives in `glicc-api.token` (git-ignored) and, for
OpenCode, in the Mac`'`s `~/.config/opencode/opencode.jsonc`. It is never committed.

```bash
TOK=$(cat glicc-api.token)                      # on the box
curl -s http://10.0.0.20:8081/v1/chat/completions \
  -H "Authorization: Bearer $TOK" -H "Content-Type: application/json" \
  -d '{"model":"glicc-model-testing","messages":[{"role":"user","content":"hi"}]}'
```
Public (once `deploy/expose-llm.sh` adds the hostname): base URL `https://llm.glicc.id/v1`,
same bearer token. Rotate the key by writing a new value into `glicc-api.token` and restarting
the server (and updating the Mac`'`s `opencode.jsonc`).
