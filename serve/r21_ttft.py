#!/usr/bin/env python3
"""R21 — TTFT (prefill) vs context length. Send prompts of growing length, read
llama-server's own prompt_ms (prefill time). Prefill is compute over all prompt
tokens, so TTFT should grow ~linearly with prompt length."""
import requests
URL = "http://10.0.0.20:8089/completion"
requests.post(URL, json={"prompt": "hi", "n_predict": 2}, timeout=60)
print(f"{'prompt tok':>10} {'prefill ms (TTFT)':>18} {'ms/1k tok':>10}")
for ntok in [128, 256, 512, 1024, 2048, 4096, 8192]:
    prompt = "word " * ntok
    r = requests.post(URL, json={"prompt": prompt, "n_predict": 1, "cache_prompt": False,
                                 "temperature": 0}, timeout=180).json()
    t = r.get("timings", {})
    pn, pms = t.get("prompt_n", ntok), t.get("prompt_ms", 0)
    print(f"{pn:>10} {pms:>18.0f} {pms/pn*1000:>10.1f}")
print("=== R21 done ===")
