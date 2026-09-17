#!/usr/bin/env python3
"""R17 — chunked prefill. A long prompt's prefill can block other requests. With
a smaller ubatch the prefill is done in more chunks, between which the scheduler
can serve concurrent decode. Measure interactive latency while big prefills run;
compare across ubatch sizes (server restarted per size)."""
import requests, time, threading, statistics as st, sys
URL = "http://10.0.0.20:8089/completion"
def interactive():
    t = time.time()
    requests.post(URL, json={"prompt": "Say hi.", "n_predict": 8, "temperature": 0}, timeout=60)
    return (time.time() - t) * 1000
def big(stop):
    while not stop.is_set():
        try:
            requests.post(URL, json={"prompt": "word " * 6000, "n_predict": 1, "cache_prompt": False}, timeout=120)
        except Exception: pass
requests.post(URL, json={"prompt": "hi", "n_predict": 2}, timeout=60)
base = [interactive() for _ in range(10)]
stop = threading.Event()
ts = [threading.Thread(target=big, args=(stop,), daemon=True) for _ in range(2)]
for t in ts: t.start()
time.sleep(1)
during = [interactive() for _ in range(15)]
stop.set()
tag = sys.argv[1] if len(sys.argv) > 1 else "?"
print(f"ubatch={tag}: interactive p50 baseline {st.median(base):.0f}ms, during big prefills {st.median(during):.0f}ms (p95 {sorted(during)[-1]:.0f})")
