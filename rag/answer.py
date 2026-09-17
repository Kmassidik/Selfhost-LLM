#!/usr/bin/env python3
"""End-to-end RAG answer: retrieve top-k chunks, ground a local model on them,
cite the source pages. The model is told to answer only from the context and to
say when the answer isn't there — the whole point of RAG over a guessing chat.
"""
import argparse, time, requests
from search import Retriever

SYS = ("You answer questions about a self-hosted-LLM field guide using ONLY the "
       "numbered context passages provided. Cite the source pages you used by their "
       "id in square brackets, e.g. [31-the-frontier-run.html]. If the answer is not "
       "in the context, say you don't find it in the guide. Be concise.")

def build_prompt(question, hits):
    ctx = "\n\n".join(f"[{i+1}] (source: {r['doc_id']}, {r['title']})\n{r['text']}"
                      for i, (_, r) in enumerate(hits))
    return f"Context:\n{ctx}\n\nQuestion: {question}"

def answer(question, retr, base_url, model, k=5, key="none"):
    t0 = time.time(); hits = retr.search(question, k); t_ret = time.time() - t0
    body = {"model": model, "temperature": 0, "max_tokens": 400,
            "messages": [{"role": "system", "content": SYS},
                         {"role": "user", "content": build_prompt(question, hits)}]}
    t1 = time.time()
    r = requests.post(base_url.rstrip("/") + "/chat/completions",
                      headers={"Authorization": f"Bearer {key}"}, json=body, timeout=120)
    r.raise_for_status()
    txt = r.json()["choices"][0]["message"]["content"]
    return txt, hits, t_ret, time.time() - t1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="+")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--base-url", default="http://10.0.0.20:8089/v1")
    ap.add_argument("--model", default="llama3")
    args = ap.parse_args()
    retr = Retriever()
    txt, hits, t_ret, t_gen = answer(" ".join(args.question), retr, args.base_url, args.model, args.k)
    print("Q:", " ".join(args.question), "\n")
    print(txt, "\n")
    print(f"--- retrieved (retrieve {t_ret*1000:.0f} ms, generate {t_gen*1000:.0f} ms) ---")
    for score, r in hits:
        print(f"  [{score:.3f}] {r['doc_id']}")

if __name__ == "__main__":
    main()
