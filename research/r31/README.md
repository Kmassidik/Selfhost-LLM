# R31 — Qwen3.6-35B-A3B vs Qwen3.8-27B: the A/B, same box, same test

Date: 2026-10-23. Question: does the MoE pick (35B total / 3B active) actually beat the dense 27B
on this box the way the active-parameter math predicted, and by how much?

**Caveat on "A/B":** this is not two servers running side by side. The box was fully cleaned between
runs (disk cleared, `models/gguf` wiped on 2026-10-23), so the original Qwen3.8-27B GSQ files no
longer exist and were not re-downloaded for this round. The comparison below is the new, live
Qwen3.6-35B-A3B measurement against the Qwen3.8-27B numbers already recorded in `research/r30/README.md`
on 2026-10-12 — same box, same cards, same `needle_test.py`, same depths where overlapping, but not
simultaneous. Treat the 27B column as **measured, 11 days earlier**, not measured now.

## Setup, this round
- Model: `unsloth/Qwen3.6-35B-A3B-GGUF`, `UD-IQ3_S` (12.7 GB on disk) plus `mmproj-F16.gguf` (0.90 GB).
- Engine: mainline `engines/llamacpp` `llama-server`, **all three cards** (0, 1, 2; `--split-mode layer`),
  one slot, `-fa on`, `--jinja`, `-ctk q4_0 -ctv q4_0`, `-c 262144`, endpoint `127.0.0.1:8081`.
- Launcher: `serving/scripts/glicc_qwen36_35b.sh` (`CTX`, `UB`, `CACHE`, `TS`, `EXTRA` env vars).
- Model id: `qwen3.6-35b-a3b` (was `qwen3.8-27b`; **OpenCode's `~/.config/opencode/opencode.jsonc`
  needs its model id and `limit.context` updated to match — not yet done as of this writing**).
- Why all three cards now, not two: the image and music services that shared cards 0–2 were stopped
  and disabled on 2026-10-23 (see handover log), so the box is now dedicated to this endpoint alone.
- Cache per token (config: 10 full-attention layers, 2 KV heads, head dim 256): 10 x 2 x 256 x 2 =
  10,240 values = 5,760 bytes at 4-bit — about a third of the 27B's 18,432 bytes/token.

## Measured (research/r30/needle_test.py, hidden fact, temperature 0, no prompt cache)

| prompt tokens | depth | prefill tok/s | decode tok/s | fact found | card memory (GiB, 0/1/2) |
|---|---|---|---|---|---|
| 48 | 0.50 | 109 | 51.9 | yes | 7.10 / 6.11 / 5.83 |
| 12,626 | 0.25 | 2,728 | 55.5 | yes | 7.10 / 6.11 / 5.83 |
| 24,445 | 0.25 | 2,728 | 59.4 | yes | 7.10 / 6.11 / 5.83 |
| 24,300 | 0.80 | 2,739 | 60.2 | yes | 7.10 / 6.11 / 5.83 |
| 48,697 | 0.50 | 2,503 | 47.8 | yes | 7.10 / 6.11 / 5.83 |
| 116,450 | 0.40 | 2,044 | 32.9 | yes | 7.10 / 6.11 / 5.83 |
| 217,313 | 0.50 | 1,594 | 21.4 | yes | 7.10 / 6.11 / 5.83 |

Card memory did not change across any run: the KV cache is pre-allocated for the full configured
`-c 262144` at server start, not grown per request. This is a real difference from the 27B round,
which needed a **different launcher config per depth** (ctx 32,768 / 65,536 / 131,072, switching
cache precision and adding `--tensor-split` partway) to avoid out-of-memory. The 35B-A3B config
above handled every depth from 48 to 217,313 tokens **without being restarted or retuned once**.

Vision: a synthetic 64x64 solid-color PNG (RGB 30/80/200) was described correctly as "Blue" in 0.36 s
wall-clock, `enable_thinking: false`, projector on GPU (no `--no-mmproj-offload` needed this time —
unlike the 27B round, which had to push the projector onto the processor to fit).

## Side by side with R30 (Qwen3.8-27B, measured 2026-10-12, not simultaneous)

| prompt tokens (closest match) | Qwen3.8-27B decode | Qwen3.6-35B-A3B decode | ratio |
|---|---|---|---|
| ~14k | 21.2 tok/s | 55.5 tok/s | 2.6x |
| ~27k (depth .25) | 18.9 tok/s | 59.4 tok/s | 3.1x |
| ~27k (depth .80) | 18.8 tok/s | 60.2 tok/s | 3.2x |
| ~54k | 14.3 tok/s | 47.8 tok/s | 3.3x |
| ~129k | 10.4 tok/s | 32.9 tok/s | 3.2x |

| prompt tokens (closest match) | Qwen3.8-27B prefill | Qwen3.6-35B-A3B prefill | ratio |
|---|---|---|---|
| ~14k | 600 tok/s | 2,728 tok/s | 4.5x |
| ~27k (depth .25) | 559 tok/s | 2,728 tok/s | 4.9x |
| ~27k (depth .80) | 552 tok/s | 2,739 tok/s | 5.0x |
| ~54k | 734 tok/s | 2,503 tok/s | 3.4x |
| ~129k | 445 tok/s | 2,044 tok/s | 4.6x |

The decode gap (2.6-3.3x) tracks closely with the active-parameter ratio predicted in the knowledge
base before this test ran (MoE 3B active vs dense 27B active, roughly an 9x parameter gap, moderated
by the shared attention/KV-read cost both architectures pay per token - so a smaller-than-9x speedup
is exactly what the earlier derivation expected, not a surprise). The prefill gap (3.4-5.0x) is new
data: R30 never isolated prefill-only scaling for the 27B's architecture against an MoE's.

## What this changes

- **Context ceiling moved**: 27B topped out at a measured 131,072 (with OOM at 163,840 and 196,608,
  needing `--tensor-split` even at 131,072). 35B-A3B held 217,313 measured tokens inside a 262,144
  config with zero OOM, zero retuning, and headroom to spare (max card use 7.10 GiB of 8.19 GiB).
- **One config, not six**: the 27B round needed six different launcher invocations across the table
  above. This round used exactly one `llama-server` process for every depth tested.
- **Still unmeasured**: a cold prefill at or near the full 262,144 ceiling (would take an estimated
  2-3 minutes at the ~1,600-2,700 tok/s band measured above, scaling down as N grows per the
  quadratic prefill term R30 derived - not re-derived fresh here).

## Open follow-ups
1. Update `~/.config/opencode/opencode.jsonc` on the Mac: model id `qwen3.6-35b-a3b`,
   `limit.context` to match whatever `CTX` the launcher is left running with.
2. The DavidAU Fable-Fusion-711 (27B dense, Q4_K_M) download is still in progress as of this
   writing - a genuine three-way comparison (27B GSQ vs 35B-A3B vs DavidAU) is not yet possible
   since the original 27B GSQ files were removed in the 2026-10-23 cleanup and were not re-fetched.
3. Handover (`STATE.md`, `OPEN.md`, log) and memory need updating to replace the 32k-ctx,
   qwen3.8-27b entries with this round's model id and 262,144-ctx config.
