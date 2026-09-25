#!/usr/bin/env python3
"""R22 (deep) — the batch-size decode frontier, the physics under R9's bottleneck.

Decode is memory-bandwidth bound: each step reads all the weights once, regardless
of how many sequences share that read. So growing the batch amortizes the same
weight-read over more sequences -> throughput climbs while wall-time-per-step
barely moves, until the GPU finally becomes compute-bound.

Force exactly DECODE tokens of generation per sequence (min=max), sweep batch
size, and measure: wall time, tokens/s, and per-sequence latency. The near-flat
wall time at small batch is exactly why serving each multi-adapter request alone
(R9 naive) throws away ~30x throughput.
"""
import time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "models/hf/Qwen2.5-0.5B-Instruct"
DECODE = 64
tok = AutoTokenizer.from_pretrained(BASE); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(BASE, dtype="bfloat16", device_map="cuda").eval()

PROMPT = tok.apply_chat_template(
    [{"role":"user","content":"Write a detailed paragraph about memory bandwidth in GPUs."}],
    tokenize=False, add_generation_prompt=True)

def bench(bs):
    prompts = [PROMPT] * bs
    enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad():
        model.generate(**enc, min_new_tokens=DECODE, max_new_tokens=DECODE,
                       do_sample=False, pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize(); dt = time.time() - t0
    return dt, bs * DECODE / dt

bench(2)  # warmup
print(f"=== R22: decode batch frontier (0.5B, {DECODE} tokens/seq forced) ===")
print(f"{'batch':>5} {'wall(s)':>8} {'tok/s':>8} {'per-seq lat(s)':>14} {'peak VRAM(GB)':>13}")
base_wall = None
for bs in [1, 2, 4, 8, 16, 32, 48, 64, 96, 128]:
    torch.cuda.reset_peak_memory_stats()
    try:
        dt, tps = bench(bs)
    except torch.cuda.OutOfMemoryError:
        print(f"{bs:>5}  OOM"); break
    peak = torch.cuda.max_memory_allocated()/1e9
    if base_wall is None: base_wall = dt
    print(f"{bs:>5} {dt:>8.2f} {tps:>8.0f} {dt:>14.2f} {peak:>13.2f}")
print(f"\nread: wall time for batch 32 vs batch 1 (should be ~similar despite 32x work),")
print("that similarity IS the batching win that naive multi-adapter serving discards.")
print("=== R22 done ===")
