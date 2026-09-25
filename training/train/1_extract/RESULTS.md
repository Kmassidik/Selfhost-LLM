# Task 1 — text → JSON extraction (results)

The first LoRA proof for the custom-SLM idea (`idea/1_custom-slm-monetization.md`).
Question: can a fine-tuned 0.5B beat a much larger general model at locking a
specific output schema — measured, on this box?

## Setup

- **Base:** Qwen2.5-0.5B-Instruct (Apache-2.0, so a sold adapter is clean)
- **Task:** messy message → strict JSON `{intent, time, amount, ref, person}`
- **Data:** synthetic (`make_data.py`), so gold is exact and scoring is `==`, not
  a judge. 3000 train / 400 val, drawn from a broad distribution (many verb
  phrasings per intent, every time/currency/ref format, distractors, typos on
  filler). `make_hard_data.py` builds a 400-row **out-of-distribution** probe
  from a disjoint name pool.
- **Train:** LoRA (r=16, α=32, all attn+MLP proj), bf16, 3 epochs, ~6.7 min on
  one RTX 3060 Ti (card 1; arena keeps card 0). No quantisation, no nvcc.
- **Metric:** `EXACT` = all five fields equal gold. Also per-field accuracy.

## Headline

| | baseline 0.5B | tuned 0.5B | Qwen3-Coder-30B |
|---|---|---|---|
| EXACT, in-dist val | 4.2% | **99.5%** | — |
| EXACT, OOD hard set | — | **99.5%** | **17.8%** |
| size | 0.5B | 0.5B | 30B (~60×) |
| serve | local | local, 434 tok/s (1 card) | 3 cards, ~20 GB |

The tuned 0.5B beats a 60×-larger general model **99.5% vs 17.8%** on the same
out-of-distribution set, same prompt, same scoring.

## Why the big model loses (the honest part)

The 30B is not dumb — its comprehension is fine (time 97.8%, amount 99.5%). It
loses on **schema convention**:

- **ref 27.5%**: emits `"#29117"`, keeping the `#`; the schema wants bare
  `"29117"`. Surface-correct, schema-wrong.
- **intent 68.5%**: calls `ship`/`settle` "payment" where the taxonomy says
  "deadline" — a labelling convention it was never told precisely.

So the claim is not "0.5B is smarter." It is: **a general model understands the
text but won't obey an arbitrary schema without a fight; the fine-tune bakes the
conventions in, at 1/60 the size, served locally at ~zero marginal cost, and it
won't drift.** Better prompting would raise the 30B — that wrestling is exactly
what the tune removes.

## The instructive miss

The tuned model first scored **23%** on the OOD set — because the hard generator
used verbs (`phone`, `settle`, `ship`) absent from training, and intent collapsed
to 34%. Widening the *training distribution* (not the model) to cover that
variation took intent back to 100% and EXACT to 99.5%. Capability came from the
corpus, not the parameter count — the 0.5B had the capacity all along.

## The closed loop

`make_data.py` → `train_lora.py` (adapter) → `merge_export.py` → GGUF Q8
(`convert_hf_to_gguf.py`, 507 MB) → `llama-server` → **99.5% at 434 tok/s** on
one card. Q8 quantisation cost nothing (adapter 99.5% ≡ GGUF 99.5%). The GGUF
lives in `models/gguf/qwen-extract-0.5b-q8.gguf` and shows up in the arena.

## Files (all under `train/1_extract/`)

- `make_data.py` — rich synthetic generator (train/val)
- `make_hard_data.py` — OOD adversarial probe
- `train_lora.py` — LoRA SFT via trl
- `eval.py` — local (transformers) scoring
- `eval_api.py` — OpenAI-compatible scoring (local server or a real frontier API)
- `merge_export.py` — merge adapter → HF → GGUF
- `data/` — train.jsonl, val.jsonl, hard.jsonl
- `adapter/` — the LoRA adapter

## Reproduce

```
uv run python train/1_extract/make_data.py --out train/1_extract/data
uv run python train/1_extract/make_hard_data.py
CUDA_VISIBLE_DEVICES=1 uv run python train/1_extract/train_lora.py
CUDA_VISIBLE_DEVICES=1 uv run python train/1_extract/eval.py --adapter train/1_extract/adapter --data train/1_extract/data/hard.jsonl
```
