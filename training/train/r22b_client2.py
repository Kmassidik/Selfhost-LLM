#!/usr/bin/env python3
"""R22b (v2) — same batched-decode frontier, but DISTINCT prompts and
cache_prompt=False, to test whether shared-prefix caching was serializing the
identical concurrent requests in v1."""
import requests, time
import concurrent.futures as cf

URL = "http://10.0.0.20:8089/completion"
N = 64

def one(i):
    p = f"Write a detailed paragraph number {i} about the history of computing and networks."
    r = requests.post(URL, json={"prompt": p, "n_predict": N, "ignore_eos": True,
                                 "temperature": 0, "cache_prompt": False}, timeout=300)
    return r.json().get("tokens_predicted", N)

print(f"{'batch':>5} {'wall(s)':>8} {'tok/s':>8} {'per-seq tok/s':>13}")
for bs in [1, 2, 4, 8, 16, 32, 64, 128]:
    one(0)  # warm
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=bs) as ex:
        toks = sum(ex.map(one, range(bs)))
    dt = time.time() - t0
    print(f"{bs:>5} {dt:>8.2f} {toks/dt:>8.0f} {toks/dt/bs:>13.1f}")
print("=== R22b v2 done ===")
