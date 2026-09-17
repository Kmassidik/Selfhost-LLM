#!/usr/bin/env python3
"""Retrieve the top-k chunks for a query by exact cosine similarity."""
import json, argparse, sys, numpy as np
from embed import encode

class Retriever:
    def __init__(self, chunks="rag/data/chunks.jsonl", emb="rag/data/emb.npy"):
        self.rows = [json.loads(l) for l in open(chunks)]
        self.emb = np.load(emb)                    # N x D, already normalised

    def search(self, query, k=5):
        q = encode([query], is_query=True)[0]      # D, normalised
        scores = self.emb @ q                      # cosine (dot of unit vectors)
        idx = np.argsort(-scores)[:k]
        return [(float(scores[i]), self.rows[i]) for i in idx]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="+")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()
    r = Retriever()
    for score, row in r.search(" ".join(args.query), args.k):
        print(f"[{score:.3f}] {row['doc_id']:32s} {row['title']}")
        print("   ", row["text"][:180].replace("\n", " "), "...")

if __name__ == "__main__":
    main()
