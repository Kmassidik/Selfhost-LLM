#!/usr/bin/env python3
"""R22b — llama.cpp batched decode frontier (the clean roofline).

Fire N concurrent requests, each forced to decode exactly 64 tokens (ignore_eos),
against llama-server. Aggregate tok/s vs batch size, on the optimized engine —
so this is true memory-bandwidth scaling, not transformers' Python overhead.
"""
import requests, time
import concurrent.futures as cf

URL = "http://10.0.0.20:8089/completion"
PROMPT = "Write a detailed paragraph about memory bandwidth in GPUs."
N = 64

def one(_):
    r = requests.post(URL, json={"prompt": PROMPT, "n_predict": N, "ignore_eos": True,
                                 "temperature": 0, "cache_prompt": True}, timeout=300)
    return r.json().get("tokens_predicted", N)

print(f"{'batch':>5} {'wall(s)':>8} {'tok/s':>8} {'per-seq tok/s':>13}")
for bs in [1, 2, 4, 8, 16, 32, 64, 128]:
    one(0)  # warm the slot/prefix
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=bs) as ex:
        toks = sum(ex.map(one, range(bs)))
    dt = time.time() - t0
    print(f"{bs:>5} {dt:>8.2f} {toks/dt:>8.0f} {toks/dt/bs:>13.1f}")
print("=== R22b done ===")
