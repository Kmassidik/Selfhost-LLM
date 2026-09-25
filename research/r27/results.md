# R27 — Typed, calibrated decisions vs free text

**Question:** does constraining a model to a typed label + a confidence-and-abstain gate cut
wrong answers, vs free-text output? (The falsifiable core of the TypeSafe AI pitch — typed
decisions with calibrated confidence, escalate when unsure.) **Model:** Qwen2.5-0.5B-Instruct.
**Task:** 30-example sentiment classification (positive/negative/neutral), incl. ambiguous ones.

## Results
| method | accuracy | parse-fail |
|---|---|---|
| A) free-text generate + parse | 66.7% | 0% (simple 1-word task parsed cleanly) |
| B) typed — constrained LM-scoring | 63.3% | **0% by construction** |
| C) typed + abstain (below confidence τ) | **up to 89.5% on answered** | see below |

**The confidence gate — calibration works:**
| τ | acc on answered | abstained | accuracy of the abstained (low-conf) set |
|---|---|---|---|
| 0.5 | 70.4% | 10% | 0% |
| 0.6 | 75.0% | 20% | 17% |
| 0.7 | **89.5%** | 37% | 18% |

**The model's low-confidence predictions are its wrong ones.** Abstaining on the ~37% it is
unsure about turns a 63% classifier into ~90% on what it answers, and routes the hard cases to
a human / bigger model. The reliability layer is real and reproducible.

## Honest caveats
- On this simple one-word task, free-text parsed fine (0% failures), so typing's *parse-safety*
  edge did not manifest — it matters more with complex/structured outputs or verbose models.
- Typing *itself* did not raise raw accuracy (63% ≈ 67%, within noise). The win is entirely the
  **calibration + abstain** layer on top.
- Small sample (30 examples) — indicative, not a benchmark.

## The lesson
"Zero hallucinations" (TypeSafe's headline) is really **typing + abstention**: constrain the
output to a valid decision, and refuse/escalate when confidence is low. It doesn't make the
model smarter — it makes it *know when to shut up*, which is exactly what autonomous automation
needs. TypeSafe's own product (Jev / RLCD) is closed and untestable; this reproduces the
concept on the box, not their model.

## Parked extension
**R27b** — proper calibration curve (reliability diagram + ECE) and grammar-constrained decoding
(SGLang) instead of LM-scoring, on a harder multi-class task. See `rnd/PARKED.md`.

## Script
`r27.py` (free-text vs typed vs typed+abstain) — archived in the box repo under `r27/`.
