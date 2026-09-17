#!/usr/bin/env python3
"""R11 demo — three users send different volumes through the metering proxy;
then read the billing ledger."""
import requests, concurrent.futures as cf

PROXY = "http://10.0.0.20:8091/v1/chat/completions"
def ask(job):
    user, prompt = job
    requests.post(PROXY, headers={"Authorization": f"Bearer {user}"},
                  json={"model": "m", "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 120, "temperature": 0.3}, timeout=120)

jobs = ([("alice", "Summarize the plot of a short mystery novel.")] * 20 +
        [("bob",   "Write a haiku about GPUs.")] * 10 +
        [("carol", "Explain TCP in one paragraph.")] * 5)
with cf.ThreadPoolExecutor(max_workers=8) as ex:
    list(ex.map(ask, jobs))

print("=== billing ledger (GET /usage) ===")
print(requests.get("http://10.0.0.20:8091/usage", timeout=30).text)
print("=== R11 done ===")
