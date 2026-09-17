# R10 — Cost per token

**Question:** what does the box actually cost per token vs a cloud API?
**Skill:** cost/token economics, TaaS pricing.

## Method
Serve Llama-3-8B Q4 on 2 cards, 16 slots; drive it to saturation with the load
harness; sample real GPU power during the run. Compute $/M tokens two ways —
marginal (electricity only) and fully-loaded (+ hardware amortization). A general
model is used (not a tuned SLM) so it generates sustained output and the tok/s is
honest and comparable to cloud pricing.

## Measured
- throughput: **398 tok/s** (batched, 16 slots, 2 cards)
- power: **197 W** GPUs + 100 W host = 297 W

## Cost (stated assumptions: $1000 box / 3 yr continuous, $0.15/kWh)

| | $/1M tokens |
|---|---|
| marginal (electricity only) | **$0.031** |
| fully-loaded (+ hw amortization, 2/3 box) | **$0.049** |
| cloud reference (Llama-3-8B class API) | ~$0.05–0.20 |

## Reading it honestly
At high utilization the $1000 second-hand box serves Llama-3-8B at **$0.03–0.05
per 1M tokens — at/below the cheap end of cloud** — while keeping inference
private. It is *not* dramatically cheaper on a general 8B model (old cards, modest
398 tok/s). The wins are privacy, zero-marginal on owned idle hardware, and — the
big one — the **custom-SLM angle**: a 0.5B specialist is ~16× smaller and packs
~50 to a card (R8), so $/task for a specialized job is far below this number.

## Pass/fail
Bar: "beat API $/task with batching." **Pass** — marginal beats cloud; fully-loaded
matches its cheap end; custom SLMs beat it more.

## Caveats
Assumes continuous high utilization (idle kills the amortization); ignores
labour/ops/reliability. The honest business case is privacy + specialization, not
a blanket "10× cheaper." Artifacts: `serve/r10_cost.sh`.
