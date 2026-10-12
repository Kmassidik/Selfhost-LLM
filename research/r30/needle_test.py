#!/usr/bin/env python3
"""R30 long-context needle test against the glicc llama-server (127.0.0.1:8081).

Builds a prompt of ~TOKENS tokens of pseudo-random filler words, hides one fact at DEPTH
(0..1) and asks for it back. Prints prefill and decode speed from the server timings.
cache_prompt is off so every run pays the full prefill.

  python3 research/r30/needle_test.py 130000 0.4
"""
import json, random, subprocess, sys, time, urllib.request

TOKENS = int(sys.argv[1]) if len(sys.argv) > 1 else 16000
DEPTH = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
MODEL = sys.argv[3] if len(sys.argv) > 3 else "qwen3.8-27b"
TOK = open("glicc-api.token").read().strip()
WORDS = ("alpha river engine copper window signal harbor lantern ridge meadow cipher orbit paper "
         "quartz timber velvet anchor bridge canyon delta ember forest glacier hollow island jungle "
         "kernel lagoon marble nectar").split()
NEEDLE = "IMPORTANT FACT: the secret code for the vault is TANGO-4471."


def req(path, body):
    r = urllib.request.Request("http://127.0.0.1:8081" + path, json.dumps(body).encode(),
                               {"Authorization": "Bearer " + TOK, "Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=3600))


random.seed(7)
para = lambda n: " ".join(random.choice(WORDS) for _ in range(n))
parts = [para(60) + " #%d." % i for i in range(int(TOKENS / 1.3) // 62)]  # ~1.3 tokens per word
parts.insert(int(len(parts) * DEPTH), NEEDLE)
text = "\n".join(parts)
n = len(req("/tokenize", {"content": text})["tokens"])
t = time.time()
d = req("/v1/chat/completions", {"model": MODEL, "max_tokens": 30, "temperature": 0, "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": text + "\n\nQuestion: what is the secret code for the vault? Answer with the code only."}]})
wall, tm = time.time() - t, d["timings"]
ans = d["choices"][0]["message"]["content"].strip()
print("tokens=%d depth=%.2f answer=%r found=%s" % (n, DEPTH, ans[:40], "TANGO-4471" in ans))
print("prefill %d tok in %.1fs = %.0f tok/s | decode %.1f tok/s | wall %.0fs" % (
    tm["prompt_n"], tm["prompt_ms"] / 1000, tm["prompt_per_second"], tm["predicted_per_second"], wall))
print(subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip().replace("\n", " | "))
