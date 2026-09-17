#!/usr/bin/env python3
"""R24 — which observable signal predicts saturation? Ramp concurrency past the
8 slots; at each level sample GPU util, active requests, queue depth (deferred),
and busy slots, and measure a probe request's TTFT (the SLO that actually matters).
The signal that crosses its threshold at the same concurrency TTFT breaks is the
one to autoscale on."""
import requests, time, threading, subprocess

BASE = "http://10.0.0.20:8089"
def metrics():
    d = {}
    for line in requests.get(BASE + "/metrics", timeout=10).text.splitlines():
        if line.startswith("llamacpp:"):
            k, v = line.split(); d[k] = float(v)
    return d
def gpu_util():
    o = subprocess.check_output(["nvidia-smi","--query-gpu=utilization.gpu","--format=csv,noheader,nounits","-i","1"])
    return float(o.decode().strip())
def heavy(stop):
    while not stop.is_set():
        try:
            requests.post(BASE + "/completion", json={"prompt": "Write a long essay about the history of computing.",
                          "n_predict": 256, "temperature": 0.7}, timeout=300)
        except Exception: pass
def probe_ttft():
    t = time.time()
    with requests.post(BASE + "/completion", json={"prompt": "Say hi.", "n_predict": 16, "stream": True},
                       stream=True, timeout=60) as r:
        for line in r.iter_lines():
            if line and b"content" in line:
                return (time.time() - t) * 1000
    return (time.time() - t) * 1000

print(f"{'conc':>4} {'TTFT ms':>8} {'gpu%':>5} {'active':>7} {'queued':>7} {'busy_slots':>10}")
for conc in [1, 2, 4, 8, 12, 16, 24, 32]:
    stop = threading.Event()
    ts = [threading.Thread(target=heavy, args=(stop,), daemon=True) for _ in range(conc)]
    for t in ts: t.start()
    time.sleep(3)
    m = metrics(); g = gpu_util(); ttft = probe_ttft()
    stop.set(); time.sleep(1.5)
    print(f"{conc:>4} {ttft:>8.0f} {g:>5.0f} {m.get('llamacpp:requests_processing',0):>7.0f} "
          f"{m.get('llamacpp:requests_deferred',0):>7.0f} {m.get('llamacpp:n_busy_slots_per_decode',0):>10.1f}")
print("=== R24 done ===")
