#!/usr/bin/env python3
"""Run a real model and time the two halves of generation separately.

Chapter 05. Producing text has two phases that share weights and share almost
nothing else: PREFILL reads the prompt (all tokens at once, in parallel) and
DECODE produces the answer (one token at a time, strictly in order). They have
unrelated speeds, and this measures both.

    python3 source/05_prefill_vs_decode.py [model-dir]

Needs torch and transformers. Uses one GPU.
"""
import os, sys, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.environ.get("SELFHOSTLLM_ROOT", "."), "models/hf/SmolLM2-360M-Instruct")
if not os.path.exists(f"{D}/config.json"):
    sys.exit(f"no model at {D}\n{__doc__}")

dev = torch.device("cuda:0")
torch.manual_seed(0)

print("loading...")
t = time.perf_counter()
tok = AutoTokenizer.from_pretrained(D)
model = AutoModelForCausalLM.from_pretrained(D, dtype=torch.bfloat16).to(dev).eval()
torch.cuda.synchronize()
print(f"  {time.perf_counter()-t:.1f}s   "
      f"{sum(p.numel() for p in model.parameters()):,} parameters on the card")
print(f"  VRAM now: {torch.cuda.memory_allocated()/1024**3:.2f} GiB\n")

PROMPT = ("The three most important things to know about running a large language "
          "model on a small graphics card are")
ids = tok(PROMPT, return_tensors="pt").input_ids.to(dev)
n_prompt = ids.shape[1]

# ── prefill: the whole prompt, one pass ─────────────────────────────
def prefill():
    with torch.no_grad():
        return model(ids, use_cache=True)

prefill()                                   # warm up kernels
torch.cuda.synchronize()
t = time.perf_counter()
out = prefill()
torch.cuda.synchronize()
t_prefill = time.perf_counter() - t
cache = out.past_key_values

print(f"=== PREFILL — {n_prompt} prompt tokens, all at once ===")
print(f"  time          {t_prefill*1e3:8.2f} ms")
print(f"  per token     {t_prefill/n_prompt*1e3:8.2f} ms")
print(f"  throughput    {n_prompt/t_prefill:8.0f} tokens/sec")

# ── what the model predicts right now ───────────────────────────────
logits = out.logits[0, -1]
probs = torch.softmax(logits.float(), dim=-1)
top = torch.topk(probs, 8)
print(f"\n  next-token distribution over {logits.shape[0]:,} tokens:")
for p, i in zip(top.values.tolist(), top.indices.tolist()):
    print(f"    {p:6.2%}  {tok.decode([i])!r}")
print(f"    ... {logits.shape[0]-8:,} more, sharing {1-top.values.sum().item():.2%}")

# ── decode: one token at a time ─────────────────────────────────────
N = 60
cur = torch.argmax(logits).view(1, 1)
times, pieces = [], [tok.decode(cur[0])]
for _ in range(N):
    torch.cuda.synchronize(); t = time.perf_counter()
    with torch.no_grad():
        o = model(cur, past_key_values=cache, use_cache=True)
    torch.cuda.synchronize(); times.append(time.perf_counter() - t)
    cache = o.past_key_values
    cur = torch.argmax(o.logits[0, -1]).view(1, 1)
    pieces.append(tok.decode(cur[0]))

import statistics as st
print(f"\n=== DECODE — {N} tokens, one at a time ===")
print(f"  median        {st.median(times)*1e3:8.2f} ms per token")
print(f"  mean          {st.mean(times)*1e3:8.2f} ms")
print(f"  throughput    {1/st.median(times):8.1f} tokens/sec")

print(f"\n=== THE COMPARISON ===")
print(f"  prefill  {n_prompt/t_prefill:8.0f} tok/s")
print(f"  decode   {1/st.median(times):8.1f} tok/s")
print(f"  prefill is {(n_prompt/t_prefill)/(1/st.median(times)):.0f}x faster per token")

# ── the KV cache, measured ──────────────────────────────────────────
def cache_bytes(c):
    """transformers has changed this API repeatedly — walk whatever it hands back."""
    total = 0
    seen = set()
    def walk(x):
        nonlocal total
        if torch.is_tensor(x):
            if id(x) not in seen:
                seen.add(id(x)); total += x.numel() * x.element_size()
        elif isinstance(x, (list, tuple)):
            for y in x: walk(y)
        elif hasattr(x, "layers"):
            for lyr in x.layers: walk(lyr)
        elif hasattr(x, "keys") and hasattr(x, "values") and not isinstance(x, dict):
            walk(x.keys); walk(x.values)
        else:
            for attr in ("key_cache", "value_cache", "keys", "values"):
                if hasattr(x, attr):
                    v = getattr(x, attr)
                    if not callable(v): walk(v)
    walk(c)
    return total

kv = cache_bytes(cache)
total_tok = n_prompt + N
print(f"\n=== KV CACHE after {total_tok} tokens ===")
print(f"  {kv/1024**2:.2f} MiB   =  {kv/total_tok:,.0f} bytes per token"
      f"  ({kv/total_tok/1024:.0f} KiB)")
cfg = model.config
formula = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * \
          (cfg.hidden_size // cfg.num_attention_heads) * 2
print(f"  formula says: 2 x {cfg.num_hidden_layers} x {cfg.num_key_value_heads} x "
      f"{cfg.hidden_size//cfg.num_attention_heads} x 2 = {formula:,} bytes/token")
print(f"  {'MATCH' if abs(kv/total_tok - formula) < 1 else 'MISMATCH'}")
for ctx in (4096, 32768, 131072):
    print(f"    at {ctx:>7,} tokens -> {formula*ctx/1024**3:6.2f} GiB")

print(f"\n=== what it wrote ===\n  {PROMPT}" + "".join(pieces))
print(f"\n  VRAM peak: {torch.cuda.max_memory_allocated()/1024**3:.2f} GiB")
