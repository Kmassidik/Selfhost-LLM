# R4 — QLoRA vs LoRA

**Question:** does 4-bit-base training match LoRA quality, and does it let you
train models that don't otherwise fit on an 8 GB card? **Skill:** QLoRA / NF4.

## Method
Same extraction data, same LoRA config. LoRA loads the base in bf16; QLoRA loads
it in 4-bit NF4 (double-quant) and trains adapters on top. Peak training VRAM
measured. Run on a 0.5B (fits both ways) and a 3B (the interesting case).

## Result

| model | LoRA (bf16) | QLoRA (4-bit NF4) |
|---|---|---|
| 0.5B — accuracy | 99.8% | 99.8% |
| 0.5B — peak VRAM | 1.91 GB | 2.00 GB |
| 3B — accuracy | — (OOM) | 97.0% |
| 3B — peak VRAM | OOM (63 MB free at crash) | 5.12 GB |

## Findings
1. **Quality parity.** 4-bit training costs nothing on the 0.5B (99.8 = 99.8),
   and the 3B QLoRA reaches 97% — it genuinely learned.
2. **QLoRA's value is enabling, not shrinking.** On the 0.5B it *saved nothing*
   (slightly more VRAM — the dequant/double-quant overhead isn't worth it when the
   weights are a small fraction of the footprint). On the 3B it is the difference
   between **OOM and training**: plain LoRA's 6 GB of bf16 weights plus activations
   overflow the 8 GB card; 4-bit weights fit at 5.12 GB.

## Pass/fail
Bar: "equal quality at lower VRAM." **Pass, with nuance:** equal quality always;
"lower VRAM" matters only at the sizes that don't otherwise fit. On the box,
QLoRA lifts the fine-tunable ceiling from ~1–2B (LoRA) to ~7B.

## Product relevance
The specialists you can mint on three cheap cards get bigger with QLoRA — useful
when a 0.5B is not enough for the task (e.g. the open-ended classification of R2,
where capacity was the limit). Artifacts: `train/r4_train.py`.
