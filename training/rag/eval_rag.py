#!/usr/bin/env python3
"""Measure the retriever: recall@k, MRR, and latency.

Build an eval set the honest way — sample chunks spread across pages, have the
local model write a question each chunk answers, then check whether that chunk's
source page comes back in the top-k. Recall@k = fraction whose source page is
retrieved; MRR rewards ranking it higher. Also times retrieval per query.
"""
import json, argparse, random, time, statistics as st, requests
from search import Retriever

def gen_questions(rows, base_url, model, key="none"):
    qs = []
    for r in rows:
        body = {"model": model, "temperature": 0.7, "max_tokens": 40,
                "messages": [{"role": "system", "content": "Write ONE specific question that the passage answers. Output only the question, no preamble."},
                             {"role": "user", "content": r["text"][:1200]}]}
        try:
            resp = requests.post(base_url.rstrip("/") + "/chat/completions",
                                 headers={"Authorization": f"Bearer {key}"}, json=body, timeout=120)
            q = resp.json()["choices"][0]["message"]["content"].strip().strip('"')
        except Exception as e:
            q = None
        if q:
            qs.append({"q": q, "gold_doc": r["doc_id"]})
    return qs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--base-url", default="http://10.0.0.20:8089/v1")
    ap.add_argument("--model", default="llama3")
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()

    retr = Retriever()
    random.seed(args.seed)
    rows = random.sample(retr.rows, min(args.n, len(retr.rows)))
    qs = gen_questions(rows, args.base_url, args.model)

    ranks, lat = [], []
    for item in qs:
        t0 = time.time()
        hits = retr.search(item["q"], k=10)
        lat.append((time.time() - t0) * 1000)
        docs = [r["doc_id"] for _, r in hits]
        ranks.append(docs.index(item["gold_doc"]) + 1 if item["gold_doc"] in docs else 0)

    def recall_at(k): return sum(1 for r in ranks if 1 <= r <= k) / len(ranks)
    mrr = sum(1.0 / r for r in ranks if r) / len(ranks)
    print(f"\n=== retrieval eval — {len(qs)} synthetic questions ===")
    print(f"recall@1 : {recall_at(1):6.1%}")
    print(f"recall@3 : {recall_at(3):6.1%}")
    print(f"recall@5 : {recall_at(5):6.1%}")
    print(f"recall@10: {recall_at(10):6.1%}")
    print(f"MRR      : {mrr:6.3f}")
    print(f"retrieve latency (ms): p50 {st.median(lat):.0f} / p95 {sorted(lat)[int(len(lat)*0.95)-1]:.0f} / max {max(lat):.0f}")
    misses = [(q["q"], q["gold_doc"]) for q, r in zip(qs, ranks) if not r]
    if misses:
        print(f"--- {len(misses)} misses (gold page never in top-10) ---")
        for q, d in misses[:5]:
            print(f"  {d}: {q[:80]}")

if __name__ == "__main__":
    main()
