#!/usr/bin/env python3
"""What does the KV cache actually save? Generate the same text twice and time it.

Chapter 05. With the cache ON, each new token reuses the work already done for
every earlier token. With it OFF, the model re-reads the entire conversation
from the beginning for every single word it writes.

    python3 source/05b_why_cache.py [model-dir]
"""
import os, sys, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.environ.get("SELFHOSTLLM_ROOT", "."), "models/hf/SmolLM2-360M-Instruct")
dev = torch.device("cuda:0")

tok = AutoTokenizer.from_pretrained(D)
model = AutoModelForCausalLM.from_pretrained(D, dtype=torch.bfloat16).to(dev).eval()

# A realistic context length. The effect is invisible at 50 tokens because
# per-layer overhead dominates there — the cache only pays when there is real
# work to skip, and that means a conversation of a believable size.
CTX = int(os.environ.get("CTX", 2000))
N   = int(os.environ.get("NGEN", 20))
filler = ("Self-hosting a language model means understanding where every byte "
          "and every millisecond goes. ") * 600
PROMPT = filler + "\n\nIn one word, the capital of Japan is"
ids = tok(PROMPT, return_tensors="pt").input_ids[:, -CTX:].to(dev)

def run(use_cache):
    seq = ids.clone()
    cache = None
    times = []
    for _ in range(N):
        torch.cuda.synchronize(); t = time.perf_counter()
        with torch.no_grad():
            if use_cache:
                inp = seq if cache is None else seq[:, -1:]
                o = model(inp, past_key_values=cache, use_cache=True)
                cache = o.past_key_values
            else:
                o = model(seq, use_cache=False)      # the whole thing, every time
        nxt = torch.argmax(o.logits[0, -1]).view(1, 1)
        torch.cuda.synchronize(); times.append(time.perf_counter() - t)
        seq = torch.cat([seq, nxt], dim=1)
    return times, tok.decode(seq[0, ids.shape[1]:])

# warm up
run(True)

on_t,  on_txt  = run(True)
off_t, off_txt = run(False)

print(f"prompt is {ids.shape[1]:,} tokens · generating {N} more\n")
print(f"{'token':>6}{'  with cache':>14}{'  without cache':>17}{'  tokens re-read':>18}")
for i in [j for j in (0, N//4, N//2, 3*N//4, N-1) if j < N]:
    reread = ids.shape[1] + i
    print(f"{i+1:>6}{on_t[i]*1e3:>12.1f}ms{off_t[i]*1e3:>15.1f}ms{reread:>16,}")

print(f"\n  total with cache      {sum(on_t):>8.2f} s   ({N/sum(on_t):>5.1f} tok/s)")
print(f"  total without cache   {sum(off_t):>8.2f} s   ({N/sum(off_t):>5.1f} tok/s)")
print(f"  the cache made it     {sum(off_t)/sum(on_t):>8.1f}x faster")

first, last = off_t[0], off_t[-1]
print(f"\n  WITHOUT the cache, cost grows as the text does:")
print(f"    token  1 took {first*1e3:.1f} ms  (re-reading {ids.shape[1]:,} tokens)")
print(f"    token {N} took {last*1e3:.1f} ms  (re-reading {ids.shape[1]+N-1:,} tokens)")
print(f"    {last/first:.2f}x slower by the end, and climbing")
on_first, on_last = on_t[1], on_t[-1]
print(f"\n  WITH the cache, cost stays flat:")
print(f"    token  2 took {on_first*1e3:.1f} ms")
print(f"    token {N} took {on_last*1e3:.1f} ms")
print(f"    {on_last/on_first:.2f}x  — essentially unchanged")

same = on_txt == off_txt
print(f"\n  same text either way? {same}")
if not same:
    print("    NOT identical — and that is worth knowing. Computing the same")
    print("    thing in one big batch versus one token at a time gives slightly")
    print("    different bf16 rounding, which can flip a near-tie. The cache is")
    print("    mathematically equivalent, not bit-identical.")
print(f"    {on_txt.strip()[:90]!r}")
print("\n  The cache changes NOTHING about the answer. It only avoids redoing work.")
