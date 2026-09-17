#!/usr/bin/env python3
"""R2 (clean) — the 99% teacher (30B) RE-LABELS the generated tickets, replacing
the noisy category-steering labels with the teacher's own judgment. This is real
knowledge distillation: the student learns the teacher's decision function.
Writes train/3_cls/distill_clean/{train,val}.jsonl."""
import requests, json, os
import concurrent.futures as cf

URL = "http://10.0.0.20:8087/v1/chat/completions"
LABELS = ["billing_payment_failure","billing_refund","account_login","account_password_reset",
 "shipping_delay","shipping_lost","product_defect","product_return","technical_bug",
 "technical_outage","feature_request","general_inquiry"]
SYS = "Classify the support ticket into exactly one category:\n"+", ".join(LABELS)+".\nReply with only the category."

def plabel(t):
    t = t.strip().lower()
    for l in LABELS:
        if t == l or t.startswith(l): return l
    for l in LABELS:
        if l in t: return l
    return None

def relabel(row):
    try:
        r = requests.post(URL, json={"model": "q", "messages": [{"role":"system","content":SYS},
                                     {"role":"user","content":row["text"]}], "max_tokens": 16, "temperature": 0}, timeout=120)
        lab = plabel(r.json()["choices"][0]["message"]["content"])
        return {"text": row["text"], "label": lab} if lab else None
    except Exception:
        return None

os.makedirs("train/3_cls/distill_clean", exist_ok=True)
changed = 0
for split in ("train", "val"):
    rows = [json.loads(l) for l in open(f"train/3_cls/distill/{split}.jsonl")]
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        out = [r for r in ex.map(relabel, rows) if r]
    changed += sum(1 for a, b in zip(rows, out) if a["label"] != b["label"])
    with open(f"train/3_cls/distill_clean/{split}.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"{split}: {len(out)} relabeled")
print(f"teacher changed {changed} labels vs the noisy category-steering (that noise is what hurt R2 v1)")
