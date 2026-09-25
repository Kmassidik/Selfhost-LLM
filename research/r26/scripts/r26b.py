#!/usr/bin/env python3
"""R26b — the deployment half: does packing ternary weights actually buy size AND speed?
Size is trivial arithmetic. Speed needs a FUSED kernel — this shows why."""
import torch, time
dev = "cuda"
K = N = 4096
W = torch.randn(N, K, device=dev)
scale = W.abs().mean()
Wt = torch.clamp(torch.round(W / scale), -1, 1)       # ternary {-1,0,1}
Wbf16 = (Wt * scale).to(torch.bfloat16)               # dequantized (what R26 simulated)
Wint8 = Wt.to(torch.int8)                             # ternary stored in int8

mb = lambda b: b / 1e6
print(f"=== SIZE of one {N}x{K} weight ({N*K/1e6:.1f}M params) ===")
print(f"  bf16 (normal)      : {mb(N*K*2):7.1f} MB   16.00 bit/w")
print(f"  int8 ternary       : {mb(N*K*1):7.1f} MB    8.00 bit/w   (2x smaller)")
print(f"  2-bit packed       : {mb(N*K*2/8):7.2f} MB    2.00 bit/w   (8x smaller)")
print(f"  1.58-bit dense trit: {mb(N*K*1.58/8):7.2f} MB    1.58 bit/w   (~10x smaller)")

def bench(fn, it=100):
    for _ in range(10): fn()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.time() - t) / it * 1e6   # microseconds

print("\n=== MATMUL latency: does storing low-bit speed it up (without a fused kernel)? ===")
for M, label in [(1, "decode  (M=1, memory-bound)"), (512, "prefill (M=512, compute-bound)")]:
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    t_bf16 = bench(lambda: x @ Wbf16.T)
    t_deq  = bench(lambda: x @ (Wint8.to(torch.bfloat16) * scale).T)  # dequant every call
    print(f"  {label:34}: bf16 cuBLAS {t_bf16:6.1f} us   |   int8-store + dequant+matmul {t_deq:6.1f} us   ({t_deq/t_bf16:.1f}x)")
print("\nlesson: storing weights small does NOT speed the matmul in stock torch — you pay a")
print("dequant back to bf16 and the matmul runs at bf16 cost (often SLOWER). The size win is")
print("free; the SPEED win needs a fused kernel that multiplies on packed weights directly.")
print("That fused kernel is exactly PrismML's fork — measured in R25b: 5.95 GB @ 32 tok/s.")
