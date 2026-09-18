#!/usr/bin/env python3
"""How much does it cost to leave a layer off the graphics card?

Chapter 11. -ngl decides how many of the model's layers live on the GPU; the
rest stay in system memory and are fetched across PCIe for every single token.
This sweeps that number from nothing to everything and reports the cost per
layer, which turns out to be almost perfectly linear — and therefore predictable.

    python3 source/11_ngl_sweep.py

Uses llama-bench, which is built for exactly this and reports prompt processing
and token generation separately.
"""
import json, os, re, subprocess, sys

B = os.environ.get("SELFHOSTLLM_ROOT", "/root/Desktop/selfhostllm")
BENCH = f"{B}/engines/llamacpp/build/bin/llama-bench"
MODEL = f"{B}/models/gguf/SmolLM2-360M-Instruct-F16.gguf"
LAYERS = 32           # this model, from its config

if not os.path.exists(BENCH):
    sys.exit(f"llama-bench not built at {BENCH}")

steps = [0, 4, 8, 16, 24, 28, 32]
print(f"SmolLM2-360M · {LAYERS} layers · one card\n")
print(f"{'-ngl':>6}{'on GPU':>9}{'in RAM':>8}{'prompt tok/s':>15}{'decode tok/s':>15}")
print("-" * 55)

rows = []
for ngl in steps:
    out = subprocess.run(
        [BENCH, "-m", MODEL, "-ngl", str(ngl), "-p", "512", "-n", "128", "-r", "2",
         "-o", "json"],
        capture_output=True, text=True, timeout=900,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="0")).stdout
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        print(f"{ngl:>6}   failed to parse llama-bench output"); continue
    pp = next((d["avg_ts"] for d in data if d.get("n_prompt")), None)
    tg = next((d["avg_ts"] for d in data if d.get("n_gen")), None)
    on, off = min(ngl, LAYERS), max(0, LAYERS - ngl)
    rows.append((ngl, on, off, pp, tg))
    print(f"{ngl:>6}{on:>9}{off:>8}{pp:>15.1f}{tg:>15.1f}")

if len(rows) >= 2:
    full = rows[-1]; none = rows[0]
    print(f"\n  all {LAYERS} layers on the card : {full[4]:.1f} tok/s")
    print(f"  all {LAYERS} layers in system RAM: {none[4]:.1f} tok/s")
    print(f"  the card is {full[4]/none[4]:.1f}x faster than system memory here")
    print(f"\n  cost of moving ONE layer off the card, measured between neighbours:")
    for i in range(1, len(rows)):
        a, b = rows[i-1], rows[i]
        d = b[1] - a[1]
        if d and a[4] and b[4]:
            print(f"    {a[0]:>2} -> {b[0]:<2} ngl  ({d} layers)  "
                  f"{a[4]:>6.1f} -> {b[4]:>6.1f} tok/s   "
                  f"{(b[4]-a[4])/d:+6.2f} tok/s per layer")
    print("\n  If that per-layer figure is roughly constant, the memory model in")
    print("  chapter 08 holds and an unmeasured -ngl can be predicted from it.")
