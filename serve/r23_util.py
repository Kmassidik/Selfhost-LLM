#!/usr/bin/env python3
"""R23 — MBU / MFU on the 3060 Ti. Measure the card's peak memory bandwidth and
peak fp16 compute, then divide the achieved decode/prefill rates (from
llama-batched-bench) by them to get utilization — the numbers that say whether
serving is leaving performance on the table."""
import torch, time

dev = "cuda"
# --- peak memory bandwidth: streaming read+write of a big buffer ---
N = 1 << 28                      # 256M fp32 = 1 GB
x = torch.empty(N, dtype=torch.float32, device=dev)
for _ in range(5): y = x * 2.0
torch.cuda.synchronize(); t = time.time()
for _ in range(50): y = x * 2.0
torch.cuda.synchronize()
bw = 2 * N * 4 / ((time.time()-t)/50) / 1e9      # read N + write N
# --- peak fp16 matmul compute ---
a = torch.randn(8192, 8192, dtype=torch.float16, device=dev)
b = torch.randn(8192, 8192, dtype=torch.float16, device=dev)
for _ in range(5): c = a @ b
torch.cuda.synchronize(); t = time.time()
for _ in range(50): c = a @ b
torch.cuda.synchronize()
tflops = 2 * 8192**3 / ((time.time()-t)/50) / 1e12

print(f"peak memory bandwidth : {bw:6.0f} GB/s  (spec 448)")
print(f"peak fp16 matmul       : {tflops:6.0f} TFLOP/s")

# --- achieved (from llama-batched-bench, qwen05b f16, 494M params) ---
PARAMS = 494e6
MODEL_GB = PARAMS * 2 / 1e9       # f16 weights ~0.99 GB
decode_tps = 240                  # batch-1 decode tok/s
prefill_tps = 22281               # batch-128 prefill tok/s (compute-bound)

mbu = (decode_tps * MODEL_GB) / bw
mfu = (prefill_tps * 2 * PARAMS / 1e12) / tflops
print(f"\ndecode (batch 1)  : {decode_tps} tok/s -> reads {MODEL_GB:.2f} GB/token")
print(f"  achieved bandwidth : {decode_tps*MODEL_GB:6.0f} GB/s   MBU = {mbu:5.0%}")
print(f"prefill (batch 128): {prefill_tps} tok/s")
print(f"  achieved compute   : {prefill_tps*2*PARAMS/1e12:6.0f} TFLOP/s   MFU = {mfu:5.0%}")
print("=== R23 done ===")
