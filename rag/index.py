#!/usr/bin/env python3
"""Embed every chunk and save the matrix. Small corpus -> a plain numpy matrix
and exact cosine search is both simplest and fastest; no vector DB needed until
the corpus is orders of magnitude bigger. Writes emb.npy (N x 384, float32)."""
import json, argparse, time, numpy as np
from embed import encode

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="rag/data/chunks.jsonl")
    ap.add_argument("--out", default="rag/data/emb.npy")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.chunks)]
    t0 = time.time()
    emb = encode([r["text"] for r in rows])
    dt = time.time() - t0
    np.save(args.out, emb)
    print(f"embedded {len(rows)} chunks -> {args.out}  shape={emb.shape}  {dt:.1f}s ({len(rows)/dt:.0f}/s)")

if __name__ == "__main__":
    main()
