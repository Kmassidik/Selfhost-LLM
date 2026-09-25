"""R28 - chasing 50 tok/s on Ternary Bonsai 2 27B (2x RTX 3060 Ti, layer split).

Two modes, one OpenAI-compatible llama-server endpoint:
  conc  - N simultaneous streams; per-stream decode tok/s and aggregate tok/s
  spec  - single stream on two prompt kinds; decode tok/s + draft acceptance

stdlib only. Writes one JSON result file.
usage: python bench.py conc|spec URL TOKEN OUT.json [label]
"""
import json, subprocess, sys, threading, time, urllib.request

PROSE = "Explain in plain words how a CPU cache works, why it speeds programs up, and what a cache miss costs. Write about 250 words."
CODE = '''Here is a Python function. Rewrite it exactly as it is, but rename the variable `total` to `acc`. Output only the code.

def summarize(orders):
    total = 0
    count = 0
    for order in orders:
        if order["status"] == "paid":
            total = total + order["amount"]
            count = count + 1
        elif order["status"] == "refunded":
            total = total - order["amount"]
    if count == 0:
        return {"total": 0, "count": 0, "average": 0}
    return {"total": total, "count": count, "average": total / count}

def by_customer(orders):
    result = {}
    for order in orders:
        key = order["customer"]
        if key not in result:
            result[key] = []
        result[key].append(order)
    return {k: summarize(v) for k, v in result.items()}
'''


def ask(url, token, prompt, n):
    body = json.dumps({
        "model": "glicc-model-testing",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": n, "temperature": 0, "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=body, headers={
        "Authorization": "Bearer " + token, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def gpu():
    q = "index,temperature.gpu,clocks.sm,utilization.gpu,clocks_throttle_reasons.active"
    out = subprocess.run(["nvidia-smi", "--query-gpu=" + q, "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    return out


def conc(url, token):
    ask(url, token, "Say hi.", 8)  # warm
    rows = []
    for n in (1, 2, 4):
        res = [None] * n

        def one(i):
            res[i] = ask(url, token, PROSE, 256)

        ts = [threading.Thread(target=one, args=(i,)) for i in range(n)]
        t0 = time.time()
        for t in ts: t.start()
        for t in ts: t.join()
        wall = time.time() - t0
        per = [r["timings"]["predicted_per_second"] for r in res]
        toks = sum(r["timings"]["predicted_n"] for r in res)
        rows.append({"streams": n, "wall_s": round(wall, 2), "tokens": toks,
                     "aggregate_tok_s": round(toks / wall, 1),
                     "per_stream_tok_s": [round(p, 1) for p in per],
                     "gpu_after": gpu()})
        print(rows[-1], flush=True)
    return rows


def spec(url, token):
    ask(url, token, "Say hi.", 8)  # warm
    rows = []
    for kind, prompt, n in (("prose", PROSE, 256), ("code-rewrite", CODE, 320)):
        for rep in range(3):
            r = ask(url, token, prompt, n)
            t = r["timings"]
            row = {"kind": kind, "rep": rep, "gen": t["predicted_n"],
                   "decode_tok_s": round(t["predicted_per_second"], 1),
                   "draft_n": t.get("draft_n"), "draft_accepted": t.get("draft_n_accepted")}
            if row["draft_n"]:
                row["accept_rate"] = round(row["draft_accepted"] / row["draft_n"], 3)
            rows.append(row)
            print(row, flush=True)
    rows.append({"gpu_after": gpu()})
    return rows


if __name__ == "__main__":
    mode, url, token, out = sys.argv[1:5]
    label = sys.argv[5] if len(sys.argv) > 5 else mode
    data = conc(url, token) if mode == "conc" else spec(url, token)
    try:
        prev = json.load(open(out))
    except Exception:
        prev = {}
    prev[label] = data
    json.dump(prev, open(out, "w"), indent=1)
