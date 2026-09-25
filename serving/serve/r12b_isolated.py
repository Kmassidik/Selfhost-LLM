#!/usr/bin/env python3
"""R12b — the fix: reserved capacity. Interactive tenant on a dedicated card,
heavy tenant hammering another. Interactive latency should stay near baseline."""
import requests, time, threading, statistics as st
HEAVY = "http://10.0.0.20:8089/v1/chat/completions"   # card 1
INT   = "http://10.0.0.20:8092/v1/chat/completions"   # card 2 (dedicated)
def interactive(url):
    t = time.time()
    requests.post(url, json={"model": "m", "messages": [{"role": "user", "content": "Say hello in one word."}],
                             "max_tokens": 16, "temperature": 0}, timeout=120)
    return (time.time()-t)*1000
def heavy(stop):
    while not stop.is_set():
        try:
            requests.post(HEAVY, json={"model": "m", "messages": [{"role": "user", "content": "Write a very long essay about the history of computing."}],
                                       "max_tokens": 512, "temperature": 0.7}, timeout=300)
        except Exception: pass
def pct(x, p): return sorted(x)[min(len(x)-1, int(len(x)*p))]
stop = threading.Event()
ts = [threading.Thread(target=heavy, args=(stop,), daemon=True) for _ in range(8)]
for t in ts: t.start()
time.sleep(4)
iso = [interactive(INT) for _ in range(20)]
stop.set()
print(f"interactive on DEDICATED card, 8x heavy load on the OTHER card:")
print(f"  p50 {st.median(iso):.0f} ms   p95 {pct(iso,.95):.0f} ms")
print("=== R12b done ===")
