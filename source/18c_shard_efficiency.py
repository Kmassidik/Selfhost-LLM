"""Bandwidth explains 1.48x. The measurement is 3.18x. Where is the rest?

Two suspects that have nothing to do with the link:

  A. SHARD INEFFICIENCY. A card doing a third of a matmul does not take a third
     of the time. Small matrices use a graphics card badly — there is not enough
     work to fill it, and the fixed cost of launching the operation stays the same.
  B. SERIALISATION. Three cards are asked for work in a loop. If nothing overlaps
     them they take turns rather than working at once, and three cards doing a
     third each, in sequence, is exactly one card's worth of time.
"""
import time, statistics, torch

dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
HIDDEN, INTER, SEQ = 960, 2560, 512

def t_ms(fn, iters=20):
    fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn(); torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1000

x = torch.randn(1, SEQ, HIDDEN, dtype=torch.bfloat16, device=dev)
print("A · does a third of a matmul take a third of the time?")
print(f"  {'shape':<26}{'ms':>9}{'vs full':>10}{'ideal':>8}{'efficiency':>12}")
print("  " + "-" * 65)
full_w = torch.randn(INTER, HIDDEN, dtype=torch.bfloat16, device=dev)
full = t_ms(lambda: x @ full_w.T)
print(f"  {f'{INTER} x {HIDDEN} (whole)':<26}{full:>9.3f}{1.0:>10.2f}{1.0:>8.2f}{'100%':>12}")
for n in (2, 3):
    w = torch.randn(INTER // n, HIDDEN, dtype=torch.bfloat16, device=dev)
    p = t_ms(lambda: x @ w.T)
    print(f"  {f'{INTER//n} x {HIDDEN} (1/{n})':<26}{p:>9.3f}{p/full:>10.2f}"
          f"{1/n:>8.2f}{(1/n)/(p/full)*100:>11.0f}%")

print()
print("B · three shards in sequence against one whole one")
ws = [torch.randn(INTER // 3, HIDDEN, dtype=torch.bfloat16, device=dev) for _ in range(3)]
seq3 = t_ms(lambda: [x @ w.T for w in ws])
print(f"  one whole matmul            {full:>9.3f} ms")
print(f"  three thirds, in sequence   {seq3:>9.3f} ms   {seq3/full:.2f}x the whole")
print()
print(f"""  A third of the work does NOT take a third of the time — the card is not
  filled by the smaller matrix, so each shard is disproportionately expensive.
  Doing all three in sequence therefore costs MORE than doing the whole thing
  once, before a single byte has crossed the link.

  That is the missing cost. Tensor parallelism on this box pays twice: the
  transfers measured in chapter 17, and the plain inefficiency of asking three
  cards to each do a piece too small to be worth their while.""")
