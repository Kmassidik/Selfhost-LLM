#!/usr/bin/env python3
"""R15 — continuous vs static batching. 48 requests with VARIABLE output lengths.
Static: process in fixed groups of 8, wait for each group's slowest before the
next (a client-imposed batch barrier). Continuous: one pool of 8 pulls the next
request whenever a slot frees. With variable lengths, continuous wins because it
never waits on stragglers."""
import requests, time, random
import concurrent.futures as cf

URL = "http://10.0.0.20:8089/completion"
random.seed(1)
N, SLOTS = 48, 8
lengths = [random.choice([16, 32, 64, 128, 256]) for _ in range(N)]

def req(npred):
    requests.post(URL, json={"prompt": "Write some text about computing.", "n_predict": npred,
                             "ignore_eos": True, "temperature": 0.7}, timeout=300)

requests.post(URL, json={"prompt": "hi", "n_predict": 2}, timeout=60)

# continuous: one pool, workers pull next as they free
t = time.time()
with cf.ThreadPoolExecutor(max_workers=SLOTS) as ex:
    list(ex.map(req, lengths))
cont = time.time() - t

# static: groups of SLOTS, barrier after each group (wait for the slowest)
t = time.time()
for i in range(0, N, SLOTS):
    with cf.ThreadPoolExecutor(max_workers=SLOTS) as ex:
        list(ex.map(req, lengths[i:i+SLOTS]))
stat = time.time() - t

print(f"{N} requests, variable output 16-256 tok, {SLOTS} slots")
print(f"continuous batching : {cont:5.1f} s")
print(f"static batching     : {stat:5.1f} s")
print(f"continuous is {stat/cont:.2f}x faster")
print("=== R15 done ===")
