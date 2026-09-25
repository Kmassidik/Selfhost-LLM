# R24 — Observability & the autoscaling signal

**Question:** which observable signal predicts saturation, so you know when to
autoscale? **Skill:** observability, autoscaling signals.

## Method
Ramp concurrency past the 8 slots; at each level sample GPU util, active requests,
queue depth (`requests_deferred`), and busy slots from llama-server `/metrics`,
and measure a probe request's TTFT (the SLO that matters).

## Result

| conc | TTFT | GPU % | active | queued |
|---|---|---|---|---|
| 1 | 29 ms | 50% | 1 | 0 |
| 4 | 46 ms | 33% | 4 | 0 |
| 8 | 4724 ms | 13% | 8 | 0 |
| 16 | 12457 ms | 15% | 8 | 15 |
| 32 | 41475 ms | 13% | 8 | 42 |

## Findings
- **GPU utilization is misleading — actively a trap.** It reads 50% idle and
  *drops to 13%* under overload (TTFT 41 s). Autoscaling on GPU% would scale DOWN
  when you must scale UP.
- **Active-slots-at-ceiling (=8) is the saturation onset signal:** TTFT breaks the
  moment slots fill (conc 8).
- **Queue depth (`requests_deferred`) is the severity signal:** 0 → 15 → 42, it
  tracks the backlog and the latency, and is the right thing to autoscale on.

## Pass/fail
Bar: "best saturation predictor named." **Pass:** scale on slot-occupancy +
queue depth; never on GPU util. Matches the production guidance (queue depth is
the reliable scaling signal; utilization misleads for GPU workloads) — demonstrated.

Artifacts: `serve/r24_signals.py`.
