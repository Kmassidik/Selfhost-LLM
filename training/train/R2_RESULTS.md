# R2 — Distillation vs LoRA (the open question from ch.33)

**Question:** classification is the task template-LoRA lost (86.8 vs the 30B's 99).
Does training the 0.5B on diverse teacher data close the gap? **Skill:** knowledge
distillation, synthetic data.

## Method
Teacher (Llama-3-8B) generates ~1560 diverse, category-steered tickets. v1 trusts
the steering category as the label. v2 has the real 99% teacher (30B) RE-LABEL
every ticket (its own judgment) — proper hard-label distillation. Student = 0.5B
LoRA, same recipe. Eval on the OOD hard set.

## Result

| classifier | macro-F1 (OOD hard) |
|---|---|
| template-LoRA (4 templates/class) | 86.8% |
| distilled, noisy steering labels (v1) | 80.9% |
| distilled, clean teacher labels (v2) | 87.8% |
| 30B teacher (reference) | 99.0% |

The 30B teacher changed **386 of 1560 labels (25%)** when re-labeling — that noise
is exactly what sank v1.

## Findings
1. **Label quality dominates diversity.** Noisy distillation *lost* to templates
   (80.9 < 86.8); cleaning the labels flipped it to a marginal win (87.8).
2. **Distillation does NOT close the gap.** 87.8 vs 99 — clean diverse data bought
   only +1pt over templates. For this open-ended 12-class task the gap is a
   capacity/knowledge limit of the 0.5B, not a data limit: hard-label distillation
   can't transfer the 30B's judgment into a model 60× smaller.

## Pass/fail
Bar: distilled ≥95% (gap mostly closed). **FAIL — honestly.** Clean distillation
reached 87.8%. The negative result is the value: it bounds the custom-SLM thesis.

## What it means for the thesis (sharpens ch.33)
- **Convention-bound tasks** (extraction, SQL): the custom SLM wins outright.
- **Open-ended semantic judgement** (this classification): neither wider data
  (ch.33) nor hard-label distillation (here) rescues the small model — rent the
  big one. The gap is real and it is capacity, not corpus.

Untested (not overclaimed past): soft-label/logit distillation, a 1.5B student.
Artifacts: `train/r2_gen.py`, `train/r2_relabel.py`, `train/3_cls/distill*/`.
