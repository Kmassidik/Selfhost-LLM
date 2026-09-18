"""Does batching change the answer? It must not.

Left-padding puts every sequence's last token in the same column, which is what
lets one cache tensor serve them all. The cost is that each sequence's REAL
position is now its column minus its own padding. Get that wrong and the model
still produces fluent text — it just believes every prompt starts at zero.

The test: send each prompt alone, then send all of them together, and compare.
Prompts of deliberately different lengths, so the padding is not uniform.
"""
import json, sys, urllib.request, concurrent.futures as cf

URL = "http://10.0.0.20:8085/v1/chat/completions"
PROMPTS = [
    "Hi",
    "The capital of France is",
    "Explain in one sentence what a KV cache is.",
    "Write a Python function that adds two numbers and returns the result.",
]

def ask(p):
    body = json.dumps({"messages": [{"role": "user", "content": p}],
                       "max_tokens": 40, "temperature": 0}).encode()
    r = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=300) as resp:
        d = json.load(resp)
    return d["choices"][0]["message"]["content"]

print("one at a time (batch of 1 each) ...")
alone = [ask(p) for p in PROMPTS]

print("all four at once (batch of 4, padding 0 to %d) ..." % 0)
with cf.ThreadPoolExecutor(max_workers=4) as ex:
    together = list(ex.map(ask, PROMPTS))

print()
ok = True
for p, a, b in zip(PROMPTS, alone, together):
    same = a == b
    ok &= same
    print(f"  {'SAME' if same else 'DIFFERS'}  {p[:44]!r}")
    if not same:
        print(f"      alone    {a[:100]!r}")
        print(f"      batched  {b[:100]!r}")
print()
print("PASS — batching does not change the answer" if ok else
      "FAIL — batched output differs from single-sequence output")
sys.exit(0 if ok else 1)
