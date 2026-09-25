#!/usr/bin/env python3
"""What happens when the model does not fit on the card?

Chapter 11. Two quantizations of the same 8-billion-parameter model: one that
fits in 7.6 GiB and one that is about 4% too large. The prediction, made from
chapter 11's offload model BEFORE either was downloaded, is that the 4% overflow
costs roughly five times the speed — because the layers that spill are fetched
across PCIe for every single token.

    python3 source/11b_model_too_big.py
"""
import json, os, subprocess, sys

B = os.environ.get("SELFHOSTLLM_ROOT", "/root/Desktop/selfhostllm")
BENCH = f"{B}/engines/llamacpp/build/bin/llama-bench"
USABLE_GB = 7.6 * 1024**3 / 1e9

CASES = [
    ("Q4_K_M", "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf", 85.1),
    ("Q8_0",   "Meta-Llama-3-8B-Instruct-Q8_0.gguf",   17.3),
]

print(f"Meta-Llama-3-8B · 32 layers · one card ({USABLE_GB:.2f} GB usable)\n")
print(f"{'quant':>8}{'file GB':>10}{'fits?':>8}{'predicted':>12}{'measured':>11}{'error':>9}")
print("-" * 60)

rows = []
for quant, fname, predicted in CASES:
    path = f"{B}/models/gguf/{fname}"
    if not os.path.exists(path):
        print(f"{quant:>8}   missing: {fname}"); continue
    gb = os.path.getsize(path) / 1e9
    for ngl, tag in ((99, None),):
        out = subprocess.run(
            [BENCH, "-m", path, "-ngl", str(ngl), "-p", "256", "-n", "64", "-r", "2", "-o", "json"],
            capture_output=True, text=True, timeout=1800,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES="0")).stdout
        try:
            data = json.loads(out)
            tg = next(d["avg_ts"] for d in data if d.get("n_gen"))
        except Exception:
            print(f"{quant:>8}{gb:>10.2f}   benchmark failed"); continue
        err = (tg - predicted) / tg * 100
        rows.append((quant, gb, tg, predicted, err))
        print(f"{quant:>8}{gb:>10.2f}{'yes' if gb < USABLE_GB else 'NO':>8}"
              f"{predicted:>10.1f}/s{tg:>9.1f}/s{err:>+8.1f}%")

if len(rows) == 2:
    fit, over = rows[0], rows[1]
    print(f"\n  the model is {over[1]/fit[1]:.2f}x larger and {fit[2]/over[2]:.1f}x slower")
    print(f"  it exceeds the card by {(over[1]-USABLE_GB)/USABLE_GB*100:.0f}% and loses "
          f"{(1-over[2]/fit[2])*100:.0f}% of its speed")
    print("\n  That asymmetry is the whole lesson: the penalty for not fitting is")
    print("  not proportional to how badly you miss. It is a cliff, not a slope.")
