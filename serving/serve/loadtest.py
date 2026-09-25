#!/usr/bin/env python3
"""Serving under load — the number an inference interview actually digs into.

Single-stream tok/s is the easy number; production is about what happens with
many callers at once. This fires a concurrency sweep at an OpenAI-compatible
endpoint, streaming each request so it can measure TTFT (time to first token),
and reports system throughput and latency percentiles at each level. The shape
to watch: throughput rises with concurrency until the server's slots saturate,
then TTFT climbs as requests queue.
"""
import argparse, time, json, statistics as st, requests
from concurrent.futures import ThreadPoolExecutor

PROMPT = ("Explain, in about 150 words, why reading model weights from memory is "
          "the bottleneck for single-stream decoding on a GPU.")

def one_request(base_url, model, max_tokens, key="none"):
    body = {"model": model, "stream": True, "temperature": 0.7, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": PROMPT}]}
    t0 = time.time(); ttft = None; toks = 0
    with requests.post(base_url.rstrip("/") + "/chat/completions",
                       headers={"Authorization": f"Bearer {key}"}, json=body,
                       stream=True, timeout=300) as r:
        for line in r.iter_lines():
            if not line:
                continue
            s = line.decode("utf-8", "ignore")
            if not s.startswith("data:"):
                continue
            s = s[5:].strip()
            if s == "[DONE]":
                break
            try:
                delta = json.loads(s)["choices"][0].get("delta", {})
            except Exception:
                continue
            if delta.get("content"):
                if ttft is None:
                    ttft = time.time() - t0
                toks += 1
    total = time.time() - t0
    return {"ttft": ttft or total, "total": total, "toks": toks}

def pct(xs, p):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(len(xs) * p))]

def run_level(base_url, model, conc, reqs, max_tokens):
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        res = list(ex.map(lambda _: one_request(base_url, model, max_tokens),
                          range(reqs)))
    wall = time.time() - t0
    toks = sum(r["toks"] for r in res)
    ttfts = [r["ttft"] * 1000 for r in res]
    e2e = [r["total"] * 1000 for r in res]
    per_stream = st.mean([r["toks"] / r["total"] for r in res if r["total"] > 0])
    return {
        "conc": conc, "sys_tps": toks / wall, "per_stream_tps": per_stream,
        "ttft_p50": pct(ttfts, .50), "ttft_p95": pct(ttfts, .95), "ttft_p99": pct(ttfts, .99),
        "e2e_p50": pct(e2e, .50), "e2e_p95": pct(e2e, .95),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://10.0.0.20:8089/v1")
    ap.add_argument("--model", default="llama3")
    ap.add_argument("--levels", default="1,2,4,8,16,32")
    ap.add_argument("--reqs-per-conc", type=int, default=4, help="requests = conc * this")
    ap.add_argument("--max-tokens", type=int, default=128)
    args = ap.parse_args()
    levels = [int(x) for x in args.levels.split(",")]

    print(f"model={args.model}  prompt~40 tok  max_tokens={args.max_tokens}\n")
    hdr = f"{'conc':>4} {'sys tok/s':>10} {'per-stream':>11} {'TTFT p50':>9} {'p95':>7} {'p99':>7} {'e2e p50':>8} {'p95':>7}"
    print(hdr); print("-" * len(hdr))
    for c in levels:
        m = run_level(args.base_url, args.model, c, c * args.reqs_per_conc, args.max_tokens)
        print(f"{m['conc']:>4} {m['sys_tps']:>10.1f} {m['per_stream_tps']:>11.1f} "
              f"{m['ttft_p50']:>8.0f}m {m['ttft_p95']:>6.0f} {m['ttft_p99']:>6.0f} "
              f"{m['e2e_p50']:>7.0f}m {m['e2e_p95']:>6.0f}")

if __name__ == "__main__":
    main()
