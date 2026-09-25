#!/usr/bin/env python3
"""Task 1 — score any OpenAI-compatible chat endpoint on the extraction set.

Same prompt and same == scoring as eval.py, but the model runs behind an HTTP
endpoint instead of locally. Point it at:
  - a local llama.cpp server (the strong-general stand-in, e.g. Qwen3-Coder-30B)
  - or a real frontier API later (--base-url + --api-key), no code change.

So the comparison is apples-to-apples: the tuned 0.5B and the big model see the
identical prompt and are graded by the identical rule.
"""
import json, argparse, re, time
from concurrent.futures import ThreadPoolExecutor
import requests

SYS = (
    "Extract the fields into JSON with exactly these keys: "
    "intent, time, amount, ref, person. "
    "intent is one of meeting, call, payment, reminder, deadline. "
    "time is 24-hour HH:MM or null. amount is a number or null. "
    "ref is a string or null. person is a string or null. "
    "Reply with only the JSON."
)
KEYS = ["intent", "time", "amount", "ref", "person"]

def extract_json(s):
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://10.0.0.20:8087/v1")
    ap.add_argument("--model", default="local")
    ap.add_argument("--api-key", default="none")
    ap.add_argument("--data", default="train/1_extract/data/hard.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data)]
    if args.limit:
        rows = rows[:args.limit]

    url = args.base_url.rstrip("/") + "/chat/completions"
    hdr = {"Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"}

    def ask(text):
        body = {
            "model": args.model,
            "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": text}],
            "temperature": 0, "max_tokens": 100,
        }
        for _ in range(3):
            try:
                r = requests.post(url, headers=hdr, json=body, timeout=120)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except Exception as e:
                last = str(e)
                time.sleep(1)
        return f"__ERR__ {last}"

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        preds = list(ex.map(ask, [r["text"] for r in rows]))
    dt = time.time() - t0

    valid = exact = errs = 0
    field_ok = {k: 0 for k in KEYS}
    for r, p in zip(rows, preds):
        if p.startswith("__ERR__"):
            errs += 1
        gold = r["gold"]
        obj = extract_json(p)
        if obj is not None and all(k in obj for k in KEYS):
            valid += 1
        obj = obj or {}
        hits = 0
        for k in KEYS:
            gv, pv = gold.get(k), obj.get(k, "__MISSING__")
            if k == "amount" and isinstance(gv, (int, float)) and isinstance(pv, (int, float)):
                ok = float(gv) == float(pv)
            else:
                ok = gv == pv
            if ok:
                field_ok[k] += 1; hits += 1
        if hits == len(KEYS):
            exact += 1

    n = len(rows)
    print(f"\n=== {args.tag or args.model} @ {args.base_url} — n={n}, {dt:.1f}s ({n/dt:.1f} req/s), errors={errs} ===")
    print(f"json_valid : {valid/n:6.1%}")
    print(f"EXACT      : {exact/n:6.1%}   <- the headline")
    for k in KEYS:
        print(f"  {k:8s}: {field_ok[k]/n:6.1%}")
    print("--- samples ---")
    for r, p in list(zip(rows, preds))[:3]:
        print("  in :", r["text"])
        print("  gold:", json.dumps(r["gold"]))
        print("  out :", p.strip().replace("\n", " ")[:160])

if __name__ == "__main__":
    main()
