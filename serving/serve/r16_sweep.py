#!/usr/bin/env python3
"""R16 (sweep) — prefix-cache speedup vs prefix length. The win should grow with
how expensive the shared prefill is."""
import requests, time, statistics as st

URL = "http://10.0.0.20:8089/completion"
BLOCK = ("You are a careful assistant that classifies and extracts structured data "
         "from customer messages according to a fixed schema and strict rules. ")

def run(prefix, cache, n=20):
    lats = []
    for i in range(n):
        t = time.time()
        requests.post(URL, json={"prompt": prefix + f"question {i}: what is item {i}?",
                                 "n_predict": 8, "cache_prompt": cache, "temperature": 0}, timeout=120)
        lats.append((time.time() - t) * 1000)
    return st.mean(lats[2:])   # warm mean

requests.post(URL, json={"prompt": "hi", "n_predict": 2}, timeout=60)
print(f"{'prefix~tok':>10} {'cache OFF ms':>12} {'cache ON ms':>12} {'speedup':>8}")
for mult in [4, 16, 48, 120]:
    prefix = BLOCK * mult + "\n\n"
    ntok = len(prefix.split()) * 4 // 3   # rough token estimate
    off = run(prefix, False); on = run(prefix, True)
    print(f"{ntok:>10} {off:>12.0f} {on:>12.0f} {off/on:>7.1f}x")
print("=== R16 sweep done ===")
