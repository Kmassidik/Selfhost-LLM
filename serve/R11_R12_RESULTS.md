# R11 + R12 — TaaS metering and multi-tenant fairness

## R11 · Token-as-a-Service metering
A proxy (`serve/r11_meter.py`) in front of the OpenAI-compatible backend reads
token usage from each response and accumulates a per-API-key ledger; `/usage`
returns the bill. Demo (three users, different volumes):

| user | reqs | prompt+completion | cost (@ $0.05/1M) |
|---|---|---|---|
| alice | 20 | 800 + 1213 | $0.000101 |
| bob | 10 | 360 + 170 | $0.0000265 |
| carol | 5 | 180 + 600 | $0.000039 |

Exact per-tenant metering + cost attribution — the TaaS layer, working. **Pass.**

## R12 · Multi-tenant fairness
Does a heavy batch tenant starve an interactive one on a shared server?

| interactive latency | p50 | p95 |
|---|---|---|
| alone | 54 ms | 155 ms |
| under 8× heavy batch (shared) | **2388 ms** | 11344 ms |
| on a **dedicated** card (heavy elsewhere) | **37 ms** | 177 ms |

**Naive slot-sharing starves the interactive tenant 44×** — llama.cpp's continuous
batching is slot-based with no priority, so small requests queue behind long batch
jobs. **Reserving capacity** (a dedicated card/slot pool) restores baseline latency
(37 ms), immune to the heavy load next door.

**Finding:** multi-tenant MaaS needs QoS. Either reserve capacity per tier
(dedicated slot pools) or add priority/preemption — otherwise one tenant's batch
job tanks everyone's latency. Matches the production guidance (priority classes,
separate pools, preemption). **Pass** — bottleneck and fix both measured.

## Where Batch B (density & monetization) lands
R8 (50 models/card) · R9 (serving is a scheduler problem, 30× naive collapse) ·
R10 ($0.03–0.05/1M tok) · R11 (metering works) · R12 (needs QoS or it starves).
Together: a mini-ixCSP is real on three cheap cards — with two hard requirements
made explicit by the measurements: **adapter-aware batching** (R9) and
**per-tenant capacity/QoS** (R12).

Artifacts: `serve/r11_meter.py`, `serve/r11_demo.py`, `serve/r12_fairness.py`,
`serve/r12b_isolated.py`.
