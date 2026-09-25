# R27 — Typed + calibrated decisions vs free text

Reproduces the falsifiable core of the TypeSafe AI pitch (typed decisions, escalate when
unsure) on Qwen2.5-0.5B. Full findings: `rnd/R27/results.md`; chip in `knowledge-base/rnd.html`.

Verdict: typed output = 0% parse-fail by construction; the real win is calibration — abstaining
on the low-confidence 37% lifts accuracy 63%->89.5%, because the model's low-confidence
predictions ARE its wrong ones. "Zero hallucinations" = typing + knowing when to abstain.

Run: `.venv/bin/python r27/scripts/r27.py` (needs box torch env + Qwen2.5-0.5B-Instruct).
