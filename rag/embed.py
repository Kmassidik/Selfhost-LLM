#!/usr/bin/env python3
"""Shared embedder — BGE-small-en-v1.5, CLS pooling + L2 normalise.

BGE wants a short instruction on the *query* side only (asymmetric retrieval);
passages are embedded bare. Cosine similarity = dot product once normalised.
"""
import numpy as np, torch
from transformers import AutoModel, AutoTokenizer

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_MODEL = None
_TOK = None

def _load(path):
    global _MODEL, _TOK
    if _MODEL is None:
        _TOK = AutoTokenizer.from_pretrained(path)
        _MODEL = AutoModel.from_pretrained(path, dtype="float16").to("cuda").eval()
    return _MODEL, _TOK

def encode(texts, path="models/hf/bge-small-en-v1.5", is_query=False, bs=64):
    model, tok = _load(path)
    if is_query:
        texts = [QUERY_PREFIX + t for t in texts]
    out = []
    for i in range(0, len(texts), bs):
        chunk = texts[i:i+bs]
        enc = tok(chunk, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
        with torch.no_grad():
            h = model(**enc).last_hidden_state[:, 0]   # CLS token
        h = torch.nn.functional.normalize(h, p=2, dim=1)
        out.append(h.float().cpu().numpy())
    return np.concatenate(out, axis=0)
