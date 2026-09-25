#!/usr/bin/env python3
"""R12 — multi-tenant fairness. Does a heavy batch tenant starve an interactive
tenant sharing the same server? Measure a small interactive request's latency
alone, then while 8 concurrent heavy (512-token) jobs saturate the slots."""
import requests, time, threading, statistics as st

URL = "http://10.0.0.20:8089/v1/chat/completions"

def interactive():
    t = time.time()
    requests.post(URL, json={"model": "m", "messages": [{"role": "user", "content": "Say hello in one word."}],
                             "max_tokens": 16, "temperature": 0}, timeout=120)
    return (time.time() - t) * 1000

def heavy(stop):
    while not stop.is_set():
        try:
            requests.post(URL, json={"model": "m", "messages": [{"role": "user", "content": "Write a very long essay about the history of computing."}],
                                     "max_tokens": 512, "temperature": 0.7}, timeout=300)
        except Exception:
            pass

def pct(x, p): return sorted(x)[min(len(x)-1, int(len(x)*p))]

print("phase 1: interactive alone")
alone = [interactive() for _ in range(20)]

print("phase 2: interactive under 8x heavy batch load")
stop = threading.Event()
ts = [threading.Thread(target=heavy, args=(stop,), daemon=True) for _ in range(8)]
for t in ts: t.start()
time.sleep(4)                    # let the heavy load fill the slots
under = [interactive() for _ in range(20)]
stop.set()

print(f"\ninteractive latency (ms):")
print(f"  alone           : p50 {st.median(alone):5.0f}  p95 {pct(alone,.95):5.0f}")
print(f"  under 8x batch  : p50 {st.median(under):5.0f}  p95 {pct(under,.95):5.0f}")
print(f"  degradation     : {st.median(under)/st.median(alone):.1f}x slower (p50)")
print("=== R12 done ===")
