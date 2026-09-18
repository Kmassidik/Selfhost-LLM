"""Big kernels overlap across cards. A stack of small ones does not. Where is the line?

19b: three 4096x4096 matmuls on three cards took 1.01x the time of one — full
overlap. 19c: three pipeline stages on three cards took 3x — none at all. The
handoff was ruled out. The remaining difference is the SHAPE of the work: one
enormous kernel against a long sequence of modest ones.
"""
import time, statistics, torch
devs = [torch.device(f"cuda:{i}") for i in range(3)]
streams = [torch.cuda.Stream(device=d) for d in devs]

def med(fn, iters=8):
    fn()
    for d in devs: torch.cuda.synchronize(d)
    ts = []
    for _ in range(iters):
        for d in devs: torch.cuda.synchronize(d)
        t0 = time.perf_counter(); fn()
        for d in devs: torch.cuda.synchronize(d)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1000

print(f"{'work per card':<34}{'1 card':>10}{'3 cards':>10}{'ratio':>8}  verdict")
print("-" * 76)
for label, N, reps in [
    ("one 4096x4096 matmul",        4096, 1),
    ("one 1024x1024 matmul",        1024, 1),
    ("50 x 1024x1024 matmuls",      1024, 50),
    ("200 x 512x512 matmuls",        512, 200),
]:
    xs = [torch.randn(N, N, dtype=torch.bfloat16, device=d) for d in devs]
    ws = [torch.randn(N, N, dtype=torch.bfloat16, device=d) for d in devs]
    def one():
        with torch.cuda.stream(streams[0]):
            for _ in range(reps): xs[0] @ ws[0]
    def three():
        for i in range(3):
            with torch.cuda.stream(streams[i]):
                for _ in range(reps): xs[i] @ ws[i]
    a, b = med(one), med(three)
    r = b / a
    v = "overlaps" if r < 1.5 else ("partial" if r < 2.4 else "SERIALISES")
    print(f"{label:<34}{a:>9.2f}ms{b:>9.2f}ms{r:>8.2f}  {v}")
    del xs, ws
    torch.cuda.empty_cache()

print("""
A single large kernel keeps a card busy long enough that the driver can issue
work to the next card while the first is still running. A long run of smaller
kernels does not: the process spends its time issuing rather than waiting, and
issuing is one thread doing one thing at a time.""")
