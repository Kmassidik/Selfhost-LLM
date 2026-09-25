#!/usr/bin/env python3
"""Chunk the corpus into passages for retrieval.

Split each page into sentences, then pack sentences into chunks of ~TARGET
characters with a one-sentence overlap so a fact spanning a boundary still lands
whole in some chunk. Each chunk keeps its source page id + title, so an answer
can cite where it came from. Writes chunks.jsonl.
"""
import json, re, argparse

def sentences(text):
    # paragraph breaks first, then sentence-ish splits inside long paragraphs
    parts = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        for s in re.split(r"(?<=[.!?])\s+", para):
            s = s.strip()
            if s:
                parts.append(s)
    return parts

def chunk_doc(text, target, overlap_sents):
    sents = sentences(text)
    chunks, cur, cur_len = [], [], 0
    i = 0
    while i < len(sents):
        s = sents[i]
        cur.append(s); cur_len += len(s) + 1
        if cur_len >= target:
            chunks.append(" ".join(cur))
            cur = cur[-overlap_sents:] if overlap_sents else []
            cur_len = sum(len(x) + 1 for x in cur)
        i += 1
    if cur and (not chunks or " ".join(cur) != chunks[-1]):
        chunks.append(" ".join(cur))
    return chunks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="rag/data/corpus.jsonl")
    ap.add_argument("--out", default="rag/data/chunks.jsonl")
    ap.add_argument("--target", type=int, default=700)
    ap.add_argument("--overlap", type=int, default=1)
    args = ap.parse_args()

    docs = [json.loads(l) for l in open(args.corpus)]
    rows = []
    for d in docs:
        for j, ch in enumerate(chunk_doc(d["text"], args.target, args.overlap)):
            rows.append({"chunk_id": f"{d['id']}#{j}", "doc_id": d["id"],
                         "title": d["title"], "text": ch})
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lens = [len(r["text"]) for r in rows]
    print(f"{len(docs)} docs -> {len(rows)} chunks -> {args.out}")
    print(f"chunk chars: min {min(lens)} / mean {sum(lens)//len(lens)} / max {max(lens)}")

if __name__ == "__main__":
    main()
