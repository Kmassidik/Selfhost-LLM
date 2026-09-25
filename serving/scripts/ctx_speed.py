#!/usr/bin/env python3
"""Decode/prefill speed vs context depth for the CPU-MoE-offloaded endpoint."""
import requests, sys
U = "http://127.0.0.1:8080/v1/chat/completions"
H = {"Authorization": "Bearer " + open("/root/model-api.token").read().strip()}
print(f"{'prompt_tok':>10} {'prefill tok/s':>13} {'decode tok/s':>13} {'prefill s':>10}")
for approx in [200, 8000, 32000, 100000]:
    filler = "The quick brown fox jumps over the lazy dog. " * (approx // 9)
    msg = filler + "\n\nReply with one short sentence."
    r = requests.post(U, headers=H, json={"model": "dalang-coder",
        "messages": [{"role": "user", "content": msg}], "max_tokens": 64, "temperature": 0},
        timeout=900).json()
    t = r.get("timings", {})
    print(f"{t.get('prompt_n',0):>10} {t.get('prompt_per_second',0):>13.0f} "
          f"{t.get('predicted_per_second',0):>13.1f} {t.get('prompt_ms',0)/1000:>10.1f}")
print("done")
