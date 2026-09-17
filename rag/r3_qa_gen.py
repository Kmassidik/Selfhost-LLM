#!/usr/bin/env python3
"""R3 — generate a KB question/answer set from the chunks (teacher = Llama-3-8B).
Short factual answers so a judge can score them. Split into train (for the
closed-book fine-tune) and a held-out test. Writes rag/data/qa_{train,test}.jsonl."""
import requests, json, re, random
import concurrent.futures as cf

URL = "http://10.0.0.20:8089/v1/chat/completions"
chunks = [json.loads(l) for l in open("rag/data/chunks.jsonl")]
random.seed(3); random.shuffle(chunks)

def gen(ch):
    prompt = ("From this passage, write ONE specific factual question and its SHORT answer "
              "(answer <= 8 words, copied from the passage). Format exactly:\nQ: <question>\nA: <answer>\n\n"
              "Passage:\n" + ch["text"][:1200])
    try:
        r = requests.post(URL, json={"model": "m", "messages": [{"role": "user", "content": prompt}],
                                     "max_tokens": 60, "temperature": 0.7}, timeout=120)
        t = r.json()["choices"][0]["message"]["content"]
        q = re.search(r"Q:\s*(.+)", t); a = re.search(r"A:\s*(.+)", t)
        if not q or not a: return None
        qa = {"q": q.group(1).strip(), "a": a.group(1).strip(), "doc": ch["doc_id"]}
        if 3 < len(qa["a"]) < 80 and len(qa["q"]) > 8:
            return qa
    except Exception:
        return None

with cf.ThreadPoolExecutor(max_workers=8) as ex:
    rows = [r for r in ex.map(gen, chunks) if r]
seen, clean = set(), []
for r in rows:
    if r["q"] in seen: continue
    seen.add(r["q"]); clean.append(r)
test, train = clean[:120], clean[120:]
for name, rs in [("train", train), ("test", test)]:
    with open(f"rag/data/qa_{name}.jsonl", "w") as f:
        for r in rs: f.write(json.dumps(r) + "\n")
print(f"QA: {len(train)} train / {len(test)} test (from {len(clean)} unique)")
for r in clean[:4]:
    print("  Q:", r["q"][:60], "| A:", r["a"][:40])
