#!/usr/bin/env python3
"""R3 — judge the 4 configs with the 30B (correct/incorrect vs gold)."""
import json, requests
import concurrent.futures as cf

URL = "http://10.0.0.20:8087/v1/chat/completions"
ans = json.load(open("rag/data/r3_answers.json"))
CONFIGS = ["base_cb", "base_rag", "tuned_cb", "tuned_rag"]

def judge(q, gold, proposed):
    p = (f"Question: {q}\nReference answer: {gold}\nProposed answer: {proposed}\n\n"
         "Is the proposed answer factually correct and consistent with the reference? "
         "Reply with exactly one word: yes or no.")
    try:
        r = requests.post(URL, json={"model": "q", "messages": [{"role":"user","content":p}],
                                     "max_tokens": 4, "temperature": 0}, timeout=120)
        return "yes" in r.json()["choices"][0]["message"]["content"].strip().lower()
    except Exception:
        return False

for cfg in CONFIGS:
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        oks = list(ex.map(lambda r: judge(r["q"], r["a"], r[cfg]), ans))
    print(f"{cfg:12s}: {sum(oks)/len(oks):6.1%} correct")
print("=== R3 judged ===")
