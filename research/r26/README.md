# R26 — Reproduce ternary QAT (how a ternary model is built)

Measured on Qwen2.5-0.5B, one 3060 Ti. Full findings: `rnd/R26/results.md`; chip in
`knowledge-base/rnd.html` (F rung).

Verdict: the 8 GB box shows the WALLS of ternary — naive rounding collapses the model
7,000,000×, group-128 scaling helps ~20× but it is still 345,000× broken, and toy-data QAT
memorizes + forgets. The RECOVERY (real QAT at scale) needs a cluster. That is why usable
sub-2-bit models are real engineering, not a trick.

## Scripts
- `scripts/r26_naive.py`  — naive ternary rounding -> the collapse.
- `scripts/r26_qat.py`    — BitLinear (ternary fake-quant + straight-through estimator), QAT.
- `scripts/r26_groups.py` — scaling-granularity sweep (scalar / row / group-128 / group-32).

Run: `.venv/bin/python r26/scripts/r26_naive.py` (needs the box torch env + Qwen2.5-0.5B).
