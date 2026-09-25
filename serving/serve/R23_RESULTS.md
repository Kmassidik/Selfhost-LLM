# R23 — MBU / MFU on the 3060 Ti

**Question:** how much of the card's bandwidth and compute does serving actually
use? **Skill:** memory-bandwidth utilization (MBU), FLOPs utilization (MFU).

## Method
Measure the card's peak memory bandwidth (streaming read+write of a 1 GB buffer)
and peak fp16 matmul, then divide the achieved decode/prefill rates (from
llama-batched-bench, qwen05b f16, 494M params) by them.

## Result
- peak memory bandwidth: **413 GB/s** (spec 448 → 92% achievable)
- peak fp16 matmul: **36 TFLOP/s**
- decode (batch 1): 240 tok/s × 0.99 GB/token = 237 GB/s → **MBU 57%**
- prefill (batch 128): 22,281 tok/s × 2·494M FLOP = 22 TFLOP/s → **MFU 61%**

## Finding
llama.cpp reaches **57% of memory bandwidth on decode** and **61% of compute on
prefill** — respectable for a consumer card, and it confirms the roofline split:
decode is bandwidth-bound (reads all weights per token), prefill is compute-bound.
Headroom: a perfect stack would be ~1.75× faster on decode, ~1.6× on prefill.

## Pass/fail
Bar: "measured for prefill & decode." **Pass** — MBU 57% (decode), MFU 61%
(prefill), against measured peaks (not spec). Artifacts: `serve/r23_util.py`.
