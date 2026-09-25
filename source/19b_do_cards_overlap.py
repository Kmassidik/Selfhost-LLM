"""Can these three cards work at the same time at all?

Chapter 19's pipeline refused to fill: six requests took six times one request.
Either stage overlap is genuinely unavailable here, or the code failed to
express it. This tests the simplest possible version of the question, with no
pipeline involved — three independent matmuls, one per card.

If three cards finish three matmuls in about the time of one, they overlap and
the pipeline code is at fault. If it takes three times as long, they do not.
"""
import time, statistics, torch

devs = [torch.device(f"cuda:{i}") for i in range(3)]
N = 4096
ws = [torch.randn(N, N, dtype=torch.bfloat16, device=d) for d in devs]
xs = [torch.randn(N, N, dtype=torch.bfloat16, device=d) for d in devs]
streams = [torch.cuda.Stream(device=d) for d in devs]

def med(fn, iters=10):
    fn()
    for d in devs: torch.cuda.synchronize(d)
    ts = []
    for _ in range(iters):
        for d in devs: torch.cuda.synchronize(d)
        t0 = time.perf_counter()
        fn()
        for d in devs: torch.cuda.synchronize(d)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1000

one = med(lambda: xs[0] @ ws[0])
def three_default():
    for i in range(3):
        xs[i] @ ws[i]
def three_streams():
    for i in range(3):
        with torch.cuda.stream(streams[i]):
            xs[i] @ ws[i]

d3 = med(three_default)
s3 = med(three_streams)

print(f"{'work':<44}{'ms':>9}{'vs one':>9}")
print("-" * 62)
print(f"{'one matmul on one card':<44}{one:>9.2f}{1.0:>9.2f}")
print(f"{'three matmuls, three cards, default stream':<44}{d3:>9.2f}{d3/one:>9.2f}")
print(f"{'three matmuls, three cards, one stream each':<44}{s3:>9.2f}{s3/one:>9.2f}")
print()
if s3 < one * 1.6:
    print("The cards DO overlap — three finish in about the time of one.")
    print("So the pipeline failing to fill is a fault in the pipeline code,")
    print("not a property of the hardware.")
else:
    print("The cards do NOT overlap. Work issued to three devices from one")
    print("Python process serialises, and the pipeline cannot fill for that")
    print("reason rather than anything about parallelism as a strategy.")
