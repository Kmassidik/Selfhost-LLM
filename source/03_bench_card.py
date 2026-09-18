"""Measure what this GPU actually does, rather than what the spec sheet says.

    python3 bench_card.py [parameter-count]

Two numbers decide almost everything in this project:
  - how fast it multiplies  (bf16 matmul throughput)
  - how fast it reads memory (bandwidth)
Both are quoted from spec sheets everywhere. Here they are measured.
"""
import torch, time

dev = torch.device("cuda:0")
print(torch.cuda.get_device_name(0))
print(f"torch {torch.__version__}\n")

def timed(fn, warmup=5, iters=20):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters

# ── matmul throughput ───────────────────────────────────────────────
print("=== bf16 matmul throughput ===")
best = 0
for n in (1024, 2048, 4096, 8192):
    a = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
    b = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
    dt = timed(lambda: torch.mm(a, b))
    tf = (2 * n**3) / dt / 1e12
    best = max(best, tf)
    print(f"  {n:>5} x {n:<5}  {dt*1e3:>8.3f} ms   {tf:>7.2f} TFLOPS")
    del a, b; torch.cuda.empty_cache()
print(f"  -> peak measured: {best:.2f} TFLOPS bf16")

# ── fp32 for comparison ─────────────────────────────────────────────
n = 4096
a = torch.randn(n, n, device=dev); b = torch.randn(n, n, device=dev)
dt = timed(lambda: torch.mm(a, b))
print(f"  fp32 {n}x{n}: {(2*n**3)/dt/1e12:.2f} TFLOPS  (for comparison)")
del a, b; torch.cuda.empty_cache()

# ── memory bandwidth ────────────────────────────────────────────────
print("\n=== memory bandwidth ===")
for mb in (256, 512, 1024):
    n = mb * 1024 * 1024 // 2          # bf16 = 2 bytes
    src = torch.randn(n, device=dev, dtype=torch.bfloat16)
    dst = torch.empty_like(src)
    dt = timed(lambda: dst.copy_(src))
    moved = n * 2 * 2                   # read + write
    print(f"  copy  {mb:>5} MiB   {dt*1e3:>8.3f} ms   {moved/dt/1e9:>7.1f} GB/s  (read+write)")
    dt = timed(lambda: src.sum())
    print(f"  read  {mb:>5} MiB   {dt*1e3:>8.3f} ms   {n*2/dt/1e9:>7.1f} GB/s  (read only)")
    del src, dst; torch.cuda.empty_cache()

# ── the number this project actually needs ──────────────────────────
# Pass a parameter count to size the test to a specific model; the default is
# SmolLM2-360M, the model ch.02 of the knowledge base opens.
import sys
PARAMS = int(sys.argv[1]) if len(sys.argv) > 1 else 361_821_120
print(f"\n=== what this means for a {PARAMS:,}-parameter model ===")
W = PARAMS * 2
buf = torch.empty(W // 2, device=dev, dtype=torch.bfloat16)
dst = torch.empty_like(buf)
dt = timed(lambda: dst.copy_(buf), warmup=3, iters=10)
bw_rw = (W * 2) / dt / 1e9
print(f"  moving {W/1e9:.2f} GB of weights took {dt*1e3:.2f} ms  ({bw_rw:.0f} GB/s read+write)")
read_only = W / (dt / 2) / 1e9 if False else None
dt2 = timed(lambda: buf.sum(), warmup=3, iters=10)
print(f"  READING {W/1e9:.2f} GB took {dt2*1e3:.2f} ms  ->  {1/dt2:.1f} tokens/sec ceiling")
print(f"  (one token requires reading every weight once)")
