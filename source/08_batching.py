#!/usr/bin/env python3
"""If decode is memory-bound, serving many requests at once should be nearly free.

Chapter 08. One token of generation requires reading every weight. If a batch of
N requests is decoded together, those same weights are read ONCE and used N
times — so total throughput should climb while each individual stream stays as
slow as it ever was.

That is a strong prediction. This measures it.

    python3 source/08_batching.py [model-dir]
"""
import os, sys, time, statistics as st, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.environ.get("SELFHOSTLLM_ROOT", "."), "models/hf/SmolLM2-360M-Instruct")
dev = torch.device("cuda:0")

tok = AutoTokenizer.from_pretrained(D)
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(D, dtype=torch.bfloat16).to(dev).eval()
P = sum(p.numel() for p in model.parameters())
WEIGHT_GB = P * 2 / 1e9
print(f"{P:,} parameters · {WEIGHT_GB:.2f} GB of weights\n")

PROMPT = "Explain in one sentence why this is"
N_GEN = 20

def run(batch):
    ids = tok([PROMPT] * batch, return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        o = model(ids, use_cache=True)
    cache = o.past_key_values
    cur = torch.argmax(o.logits[:, -1], dim=-1).unsqueeze(-1)
    times = []
    for _ in range(N_GEN):
        torch.cuda.synchronize(); t = time.perf_counter()
        with torch.no_grad():
            o = model(cur, past_key_values=cache, use_cache=True)
        torch.cuda.synchronize(); times.append(time.perf_counter() - t)
        cache = o.past_key_values
        cur = torch.argmax(o.logits[:, -1], dim=-1).unsqueeze(-1)
    return st.median(times)

run(1)   # warm up

print(f"{'batch':>6}{'ms/step':>10}{'per-stream':>13}{'TOTAL':>12}{'vs batch 1':>12}{'GB read/token':>15}")
print("-" * 70)
base = None
for b in (1, 2, 4, 8, 16, 32, 64):
    try:
        dt = run(b)
    except torch.cuda.OutOfMemoryError:
        print(f"{b:>6}   out of memory"); break
    per_stream = 1 / dt
    total = b / dt
    if base is None: base = total
    # the weights are fetched once per STEP, not once per token
    gb_per_token = WEIGHT_GB / b
    print(f"{b:>6}{dt*1e3:>10.1f}{per_stream:>11.1f}/s{total:>10.1f}/s"
          f"{total/base:>11.1f}x{gb_per_token:>14.3f}")
    torch.cuda.empty_cache()

print(f"\n  A single stream never gets faster — it cannot. Each token still waits")
print(f"  for the whole {WEIGHT_GB:.2f} GB to arrive.")
print(f"  But the SAME arrival now serves every request in the batch, so the")
print(f"  bytes-read-per-token falls as 1/batch and total throughput climbs.")
print(f"\n  peak VRAM {torch.cuda.max_memory_allocated()/1024**3:.2f} GiB")
