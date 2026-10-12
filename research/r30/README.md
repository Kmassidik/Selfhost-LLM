# R30 - a coding + vision model on the shared box (Qwen3.8-27B, 3.5-bit)

Date: 2026-10-12. Question: what is the best vision-capable coder this box can serve to one user,
and how much context does it get? The box is shared (card 0 = music engine, card 1 = image service,
stopped; card 2 = Bonsai, stopped for this run).

## Setup
- Model: `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF`, `IQ3_S` (11.77 GB, 3.5 bits per weight) plus the
  separate `mmproj-Qwen3.8-27B-BF16.gguf` (0.93 GB) for images.
- Engine: mainline `engines/llamacpp` `llama-server`, cards 1 and 2 (`--split-mode layer`), one slot,
  `-fa on`, `--jinja`, endpoint `127.0.0.1:8081` behind `llm.glicc.id`.
- Launcher: `serving/scripts/glicc_qwen38_27b.sh` (`CTX`, `UB`, `EXTRA` environment variables).
- Cache per token (config: 64 layers, every 4th is full attention, 4 KV heads, head dim 256):
  16 x 4 x 256 x 2 = 32,768 values = 18,432 bytes at 4-bit, 34,816 at 8-bit.

## Measured (research/r30/needle_test.py, hidden fact, temperature 0, no prompt cache)
| prompt tokens | ctx / cache / ub | prefill | decode | fact found | card memory (GiB) |
|---|---|---|---|---|---|
| 27 (short) | 32,768 / q8_0 / 256 | - | 24.7 tok/s | n/a | 7.16 / 7.16 |
| 14,220 (depth .25) | 32,768 / q8_0 / 256 | 600 tok/s | 21.2 | yes | 7.16 / 7.16 |
| 27,348 (depth .25) | 32,768 / q8_0 / 256 | 559 tok/s | 18.9 | yes | 7.16 / 7.16 |
| 27,221 (depth .80) | 32,768 / q8_0 / 256 | 552 tok/s | 18.8 | yes | 7.16 / 7.16 |
| 54,347 (depth .50) | 65,536 / q8_0 / 256 | 734 tok/s | 14.3 | yes | 7.58 / 7.77 |
| 129,180 (depth .40) | 131,072 / q4_0 / 128, `--tensor-split 55,45 --no-mmproj-offload` | 445 tok/s (290 s) | 10.4 | yes | 7.64 / 7.57 |

Images work (synthetic 128 px and 256 px pictures, described correctly); with the projector on the
processor an image request took 6.2 s.

## What did not fit
- 65,536 / q8_0 / ub 512 on two cards: out of memory on card 2 while loading (first attempt).
- 196,608 and 163,840 / q4_0: out of memory allocating compute buffers.
- 131,072 / q4_0 / ub 128 with the default split: out of memory; `--tensor-split 55,45` fixed it.
- A restart within a second of the previous server exiting fails (its memory is not freed yet); the
  launcher now sleeps 6 s after killing the old process.

## Derived (not measured)
- Decode time per token fits `0.040 + 4.3e-7 x depth` s (points above). Extrapolated: ~6.5 tok/s at
  262k, ~2 tok/s at 1M.
- Prefill time fits `a*N + b*N^2/2` with a = 0.72 ms, b = 1.19e-8 s per token of depth: ~10 min for
  262k and ~1.9 h for a cold 1M prompt.
- 1M tokens of cache = 18.4 GB at 4-bit: does not fit three 8 GiB cards next to the weights.

## OpenCode finding
The client config had `limit: {context: 32768, output: 32768}`. OpenCode reserves the output size from
the window, so usable input was about 0 and every turn compacted. Its own baseline prompt is ~14.7k
tokens (12,691 input + 2,048 cached on the first turn). Fix: `context 131072, output 8192`.

## Not measured
Answer quality beyond the needle, a second turn with prompt-cache reuse at depth, quality of the
3.5-bit file against the 4-bit file on real tasks (ch.53: needs ~1,000 tasks for a +-3 point margin).
