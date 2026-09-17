# R7 — Cross-task quant sensitivity

**Question:** does quantization break harder tasks earlier than extraction?
**Skill:** task-aware quantization quality-loss.

## The detour that became a finding (R7b)

First attempt quantized via GGUF k-quants (as R1 did for extraction). It broke:
served SQL/CLS scored ~39% / ~58% even at **f16** — near the *base* model — while
the merged HF model served at **99% / 86.8%**. The merge is verified good (HF is
faithful); the **GGUF conversion/serving path degrades the harder tuned models**,
which extraction was robust enough to hide.

- SQL merged HF (transformers): **99%** exec-match.
- SQL merged GGUF f16 (llama.cpp): **39%** — simple queries fine, joins/aggregates fail.
- Tied embeddings identical across all three tasks, so not the differentiator.

**Implication (MaaS):** some custom SLMs must be served via safetensors (vLLM /
transformers), not GGUF/llama.cpp. Root cause (conversion vs kernel numerics) is
tagged **R7b** as a follow-up (compare logits on one prompt).

## R7 proper — faithful path (bitsandbytes int8 / nf4, adapter on top)

| task | bf16 | int8 | nf4 (4-bit) |
|---|---|---|---|
| extract (rigid schema) | 99.5% | 99.5% | 98.8% |
| sql (structured) | 99.5% | 98.5% | 92.5% |
| cls (open-ended) | 86.8% | 87.9% | 73.9% |

**int8 is free** for all three. **4-bit (nf4) degrades in proportion to task
openness:** extraction −0.7pt, SQL −7pt, classification −12.9pt. The rigid-schema
task is quant-robust; the semantic-judgement task is quant-fragile.

## Pass/fail
Bar: "which task breaks first, why." **Pass.** Classification breaks first under
4-bit; extraction is most robust; the ordering tracks how open-ended the task is —
because open-ended tasks live nearer the decision boundary, where 4-bit weight
noise flips more tokens.

## Product relevance
For MaaS packaging: rigid-schema specialists ship at 4-bit (or GGUF Q2) with
near-zero loss = maximum density; open-ended specialists need int8 (or a
safetensors serve) to stay accurate. Compression budget is task-dependent.

Artifacts: `train/*/eval*.py --quant int8|nf4`, `run_r7_bnb.sh`.
