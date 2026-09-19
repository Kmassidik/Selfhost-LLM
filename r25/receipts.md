# R25 — Ternary Bonsai 2 27B: receipts

## A. Third-party receipt — RTX 3060 **12 GB** (quoted, NOT measured on our box)

Source: community/PrismML report. Our box has 3060 **Ti 8 GB**, so this 12 GB sheet is
a reference row, not something we reproduced. Kept verbatim as the "12 GB row" of the
small-GPU guide.

**Speed by depth** (live server, thinking on, real sessions)
| context depth | decode tok/s |
|---|---|
| 7k  | 24.4 |
| 12k | 21.9 |
| 35k | 17.8 |
| 77k | 13.0 |

**Context memory cost** (q4 KV, flash attention, full native window resident)
| window | resident VRAM |
|---|---|
| 64k  | 7.3 GB |
| 128k | 8.8 GB |
| 192k | 10.2 GB |
| 262k | 11.7 GB (0.6 GB to spare — whole native window on a 12 GB card) |

- **~1.47 GB per 64k of context** → 327k would not fit.
- A **41,312-token build session** (35k→77k depth) averaged **15.0 tok/s over 46 min**.
- Prefill **295 tok/s @ 2k**, **243 tok/s @ 35k**; first token **0.6 s**; fresh decode **26.1 tok/s**.
- Card pinned **149/150 W**, **78 °C**, fan 80% — **power-bound, not heat-bound**. **0.158 tok/s per watt** at the fresh end.

**Setup:** model ternary Bonsai 2 27B, PTQ1_0, 1.75 bpw, 5.95 GB, base Qwen 3.8 27B, Apache-2.0.
Runtime: PrismML llama.cpp fork, prebuilt CUDA 12.4 binary. Serve: full 262k window resident,
q4 KV, flash attention, one slot, 11.7/12 GB.

Bonsai 1 (July) context: 3.9 GB 1-bit file at 42 tok/s on a 3060 Ti — faster card, smaller
file, so same-card comparison isn't on the table until the 8 GB test. What changed in v2:
base Qwen 3.8 (was 3.6), retention 98.2% on their suite (was 95%), full 262k on 12 GB.

## B. Our measured numbers

**R25a — Mac (M2 Pro, MLX 2-bit pack, 8.61 GB)**
- bit-identical pack (sha256 match); 129 residual writers (exact); refusal component → 1e-7…1e-9.
- 0 weights modified (hash identical pre/post). ~10–12 tok/s. Quality preserved on benign set.
- Uncensoring: flipped 2/2 benign refusals (own-door lock-pick, own-car hotwire).

**R25b — Box (RTX 3060 Ti 8 GB, GGUF PTQ1_0 via PrismML fork, CUDA)**
- On disk 5.95 GB; engine reports `PTQ1_0 - 1.75 bpw ternary (group 128)`.
- Runs on ONE 8 GB card, `-ngl 99`: **32.3 tok/s decode**, **102 t/s prompt** (short ctx).

**R25b — the 8 GB row (measured, q4 KV + flash attn, one slot)**
- **Context ceiling: 64k window fits (7.79 / 8 GB); 96k+ OOM.** The 8 GB card cannot hold
  the KV for the full 262k native window (that needs the 12 GB card).
- Speed by depth:
  | depth | decode | prefill | TTFT |
  |---|---|---|---|
  | 2.3k | 31.2 tok/s | 324 tok/s | 7.0 s |
  | 7.2k | 29.1 tok/s | 333 tok/s | 21.6 s |
  | 30.5k | 21.8 tok/s | 306 tok/s | 99.8 s |
- Same-card-class comparison vs their 3060 12 GB sheet: our 3060 Ti is ~15–20% faster per
  token (29.1 vs 24.4 @ ~7k; 21.8 vs 17.8 @ ~35k) but caps at 64k vs 262k window.
  The honest trade: **faster card, smaller window.**

**R25b — 2× 8 GB layer split (measured): the full 262k window at full speed**
- `-sm layer` across 2 cards, `-c 262144`, q4 KV, flash attn: **LOADS the whole 262k window.**
  Per-card VRAM 7.67 / 7.77 GB (combined ~15.4 GB). **decode 32.9 tok/s**, prompt 96 tok/s.
- So two 8 GB cards hold the entire native window at single-card speed — no RAM-offload penalty.
  Ternary weights (5.95 GB, split) + hybrid-attention KV (~6 GB @ 262k) + compute fit in 16 GB.
- Ladder (all measured): 1 card = 64k @ ~31 tok/s; RAM offload = 262k @ ~5–8 tok/s (est.);
  **2 cards layer split = 262k @ 32.9 tok/s** (the sweet spot). Tensor/row split fails on this
  PCIe-only box (no peer access, per R19) — layer split is the only viable multi-card mode.
- Prebuilt CUDA binary would NOT run (Xeon E5-2665 has no AVX2 → illegal instruction); must
  build the PrismML fork from source so cmake targets the box's actual CPU ISA.

## C. Why the full 262k window fits on 12 GB (the mechanism)

Three multipliers, not magic:
1. **Tiny model** — 5.95 GB ternary (1.75 bpw), not ~54 GB fp16.
2. **q4 KV cache** — 4-bit KV quantization, ~¼ of fp16 KV.
3. **Hybrid attention** — config shows **64 layers = 48 linear-attention + 16 full-attention**
   (full every 4th). Only the **16 full-attention layers** keep a *growing* KV cache; the 48
   linear-attention layers carry a fixed-size recurrent state. So KV grows at ~1.47 GB/64k
   instead of the ~6 GB/64k a dense-attention 27B would need. A dense 27B at 262k would need
   ~17 GB of KV and could not fit.

Internal consistency of the 12 GB sheet: deltas 7.3→8.8→10.2→11.7 GB are ~1.47 GB per 64k;
262k+64k ≈ 13.2 GB > 12 GB, matching "327k would not fit." Numbers are self-consistent and
physically plausible. **Verdict: credible.**

## D. Pending — the 8 GB test (our card, box-native)
The same-card 8 GB comparison the community says "isn't on the table yet" — we CAN run it:
measure the max context window that fits on our 3060 Ti 8 GB with PTQ1_0 + q4 KV, and the
speed-by-depth curve, to produce the honest 8 GB row alongside their 12 GB row.
