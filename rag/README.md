# rag/ — retrieval-augmented generation over the knowledge base

Closing the applied-LLM gap: a working, measured RAG system whose corpus is this
guide's own 35 pages. Ask a question, retrieve the relevant chunks, answer with a
local model grounded on them, cite the source pages.

## Pipeline

```
corpus.jsonl  (35 KB pages, extracted from the KB HTML)
   -> ingest.py   sentence-pack into 677 chunks (~750 chars, 1-sentence overlap)
   -> index.py    embed with BGE-small-en-v1.5 (384-dim, CLS pool, normalise) -> emb.npy
   -> search.py   query -> top-k by exact cosine (numpy matmul; no vector DB needed here)
   -> answer.py   ground a local model (Llama-3-8B) on the top-k, cite pages
```

## Measured (on this box)

| | value |
|---|---|
| corpus | 35 pages -> 677 chunks |
| embed the whole corpus | 2.4 s (279 chunks/s) on one card |
| retrieval recall@1 | 75.0% |
| retrieval recall@3 | 88.3% |
| retrieval recall@5 | 93.3% |
| retrieval MRR | 0.825 |
| retrieve latency | p50 14 ms / p95 16 ms (embed query + cosine over 677) |
| end-to-end answer | ~2.4 s (retrieve + Llama-3-8B generate) |

Eval set: 60 questions the local model wrote from sampled chunks; recall@k =
fraction whose source page returns in top-k. The few misses are vague generated
questions ("what is the architecture of the engine?") that match many pages —
eval noise more than retrieval failure.

## Why these choices

- **BGE-small (33M):** small, fast, strong on retrieval; asymmetric — a short
  instruction on the query side only. Big enough to be good, small enough that
  embedding the corpus is 2.4 s.
- **numpy matrix, not a vector DB:** at 677 chunks exact cosine is 14 ms and
  simplest. A DB (FAISS/pgvector) earns its keep at ~10^5+ chunks, not here —
  and saying so is the honest engineering call.
- **grounded prompt:** the model answers only from the passages and cites pages,
  which is the point of RAG over a chat that guesses.

## Files (rag/)

`embed.py` (shared embedder) · `ingest.py` (chunk) · `index.py` (embed) ·
`search.py` (retrieve) · `answer.py` (retrieve + generate + cite) ·
`eval_rag.py` (recall@k, MRR, latency) · `data/` (corpus, chunks, emb.npy)

## Reproduce

```
uv run python rag/ingest.py && CUDA_VISIBLE_DEVICES=1 uv run python rag/index.py
# serve an answerer:
CUDA_VISIBLE_DEVICES=2 engines/llamacpp/build/bin/llama-server \
  -m models/gguf/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --host 10.0.0.20 --port 8089 -ngl 99 -a llama3 &
CUDA_VISIBLE_DEVICES=1 uv run python rag/answer.py "why cant vLLM run a GGUF model?"
CUDA_VISIBLE_DEVICES=1 uv run python rag/eval_rag.py --n 60
```
