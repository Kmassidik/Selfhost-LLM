# train/ — fine-tuning small models on the box

LoRA experiments testing the custom-SLM idea (`idea/1_custom-slm-monetization.md`):
can a fine-tuned 0.5B beat a much larger general model on a narrow task —
measured, on three second-hand 8 GB cards?

Base for all three: **Qwen2.5-0.5B-Instruct** (Apache-2.0). Strong general
stand-in: **Qwen3-Coder-30B-A3B** (~60× the size), served locally via llama.cpp
through the same OpenAI-compatible endpoint the eval scripts use — so a real
frontier API drops in with `--base-url` + `--api-key`, no code change. Every
task is scored on an **out-of-distribution hard set** (held-out phrasings the
model never trained on). LoRA trains in 3–7 min on one card.

## The three results

| task | metric | baseline 0.5B | **tuned 0.5B** | 30B (60×) | winner |
|---|---|---|---|---|---|
| 1 · extraction | exact-match (strict schema) | 4.2% | **99.5%** | 17.8% | **tuned, decisive** |
| 2 · text→SQL | execution (forgiving surface) | 36.5% | **99.5%** | 89.5% | **tuned, modest** |
| 3 · classification | macro-F1 (open-ended semantic) | 32.0% | 86.8% | **99.0%** | **30B** |

## What it actually says (the honest synthesis)

The fine-tuned SLM's advantage is **not universal** — it tracks how much the task
is about obeying a convention vs. understanding the world:

- **Wins decisively** when the task is locking an *arbitrary output convention*
  the big model won't obey without a fight (extraction: it wrote `"#29117"`
  where the schema wanted bare `29117`; it guessed the intent taxonomy). 99.5%
  vs 17.8%.
- **Wins modestly** when the metric *forgives surface form* (SQL execution
  compares result rows, not query text), so the big model's competence shows
  through. 99.5% vs 89.5%.
- **Loses** when the task is *open-ended semantic judgement over novel inputs*
  (routing unseen phrasings across 12 adjacent categories). The 30B's world
  knowledge generalises better than a 0.5B trained on narrow templates. 86.8%
  vs 99.0%.

So the sellable case for a custom SLM is the **narrow, format- or
convention-bound task**, hosted locally at ~zero marginal cost (tuned model:
507 MB GGUF, 434 tok/s on one 3060 Ti). For open-ended semantic work, rent the
big model — or, per the recurring lesson, **widen the training data**: task 3's
gap is a coverage gap (4 templates/class), not a capacity limit, and richer
data would close much of it. Capability is training, not size.

## The instructive miss (task 1)

Task 1 first scored 23% on its OOD set — the hard generator used verbs absent
from training and intent collapsed to 34%. Widening the *training distribution*
(not the model) took it to 99.5%. The failure is in the git history on purpose.

## Layout

```
train/
  1_extract/   text -> strict JSON     make_data.py make_hard_data.py train_lora.py eval.py eval_api.py merge_export.py RESULTS.md
  2_sql/        text -> SQL             build_db.py make_data_sql.py train_lora_sql.py eval_sql.py  (sql.db, schema.sql)
  3_cls/        ticket routing          make_data_cls.py train_lora_cls.py eval_cls.py
```

Each `eval*.py` runs locally (`--adapter`) or against any OpenAI-compatible
endpoint (`--base-url`). Exported GGUF: `models/gguf/qwen-extract-0.5b-q8.gguf`
(task 1), visible in the arena.
