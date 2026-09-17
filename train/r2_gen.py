#!/usr/bin/env python3
"""R2 — distillation: a larger model (teacher) generates a large, DIVERSE labeled
ticket set; the 0.5B student trains on it instead of the 4 narrow templates. Tests
whether diverse teacher data closes the generalization gap template-LoRA lost.
Writes train/3_cls/distill/{train,val}.jsonl."""
import requests, json, os, random
import concurrent.futures as cf

URL = "http://10.0.0.20:8089/v1/chat/completions"
LABELS = ["billing_payment_failure","billing_refund","account_login","account_password_reset",
 "shipping_delay","shipping_lost","product_defect","product_return","technical_bug",
 "technical_outage","feature_request","general_inquiry"]
PER = 130

def gen_one(cat):
    prompt = (f"Write ONE short, realistic customer-support message (1-2 sentences) that a "
              f"support system should route to the category '{cat}'. Vary the wording and tone "
              f"naturally; do NOT mention the category name or use obvious keywords. Output only the message.")
    try:
        r = requests.post(URL, json={"model": "m", "messages": [{"role": "user", "content": prompt}],
                                     "max_tokens": 60, "temperature": 1.0}, timeout=120)
        t = r.json()["choices"][0]["message"]["content"].strip().strip('"').replace("\n", " ")
        return {"text": t, "label": cat}
    except Exception:
        return None

jobs = [c for c in LABELS for _ in range(PER)]
random.shuffle(jobs)
with cf.ThreadPoolExecutor(max_workers=8) as ex:
    rows = [r for r in ex.map(gen_one, jobs) if r]

seen, clean = set(), []
for r in rows:
    k = r["text"].lower()
    if len(r["text"]) < 10 or k in seen:
        continue
    seen.add(k); clean.append(r)
random.shuffle(clean)
n_val = min(240, len(clean)//6)
val, train = clean[:n_val], clean[n_val:]
os.makedirs("train/3_cls/distill", exist_ok=True)
for name, rs in [("train", train), ("val", val)]:
    with open(f"train/3_cls/distill/{name}.jsonl", "w") as f:
        for r in rs:
            f.write(json.dumps(r) + "\n")
print(f"distill set: {len(train)} train / {len(val)} val (from {len(rows)} generated, {len(clean)} unique)")
from collections import Counter
print("per-class:", dict(Counter(r["label"] for r in clean)))
for r in clean[:4]:
    print("  ", r["label"], "->", r["text"][:70])
