#!/usr/bin/env python3
"""R16 — prefix-cache win. Many requests share a long system prompt (schema /
labels / RAG context) and differ only in a short suffix. With prompt caching the
shared prefix is prefilled once and reused; without it, every request re-prefills
the whole thing. Measure per-request latency (short generation, so it's
prefill-dominated), cache on vs off, over a sequential same-prefix stream."""
import requests, time, statistics as st

URL = "http://10.0.0.20:8089/completion"
# a long shared prefix (~400-500 tokens): a verbose instruction block, repeated
BLOCK = ("You are a careful assistant that classifies and extracts structured data "
         "from customer messages according to a fixed schema and a strict set of rules "
         "that must always be followed exactly and never deviated from under any "
         "circumstance whatsoever. ")
PREFIX = BLOCK * 12 + "\n\nNow answer this question briefly. Question: "

def run(cache):
    lats = []
    for i in range(30):
        p = PREFIX + f"what is item number {i} called?"
        t = time.time()
        requests.post(URL, json={"prompt": p, "n_predict": 8, "cache_prompt": cache,
                                 "temperature": 0}, timeout=120)
        lats.append((time.time() - t) * 1000)
    return lats

# warm one call so the model is ready
requests.post(URL, json={"prompt": "hi", "n_predict": 2}, timeout=60)
on = run(True)
off = run(False)
plen = len(PREFIX.split())
print(f"shared prefix ~{plen} words; 30 sequential requests, 8-token gen")
print(f"cache ON : first {on[0]:.0f} ms, rest mean {st.mean(on[1:]):.0f} ms")
print(f"cache OFF: first {off[0]:.0f} ms, rest mean {st.mean(off[1:]):.0f} ms")
print(f"prefix-cache speedup on warm requests: {st.mean(off[1:])/st.mean(on[1:]):.1f}x")
print("=== R16 done ===")
