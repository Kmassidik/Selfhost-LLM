#!/usr/bin/env python3
"""Why a grid is really a line: strides, and the free transpose.

Chapter 02. Memory takes one address, so any grid is flattened and a stride
says how to walk it. The proof is that transposing costs nothing — same memory,
swapped strides, zero bytes moved.

    python3 source/02c_strides.py
"""
import torch, time

print("=== a 3x4 grid, and where its numbers actually sit ===")
a = torch.arange(12, dtype=torch.float32).reshape(3, 4)
print(a)
print(f"\n  shape   {tuple(a.shape)}")
print(f"  stride  {a.stride()}   <- steps to move one place in each direction")
print(f"  memory  {a.flatten().tolist()}   <- what is really there: ONE line")

print("\n=== the mapping, done by hand ===")
rows, cols = a.shape
for (i, j) in [(0,0), (0,3), (1,0), (2,1)]:
    idx = i*cols + j
    print(f"  element [{i},{j}] = {a[i,j].item():4.0f}   lives at position {idx:2}   (i x {cols} + j)")

print("\n=== transpose: does anything move? ===")
b = a.T
print(f"  original  shape {tuple(a.shape)}  stride {a.stride()}  data at {a.data_ptr()}")
print(f"  transposed shape {tuple(b.shape)}  stride {b.stride()}  data at {b.data_ptr()}")
print(f"  same memory? {a.data_ptr() == b.data_ptr()}")
print(f"  bytes copied: 0 — only the stride was swapped, {a.stride()} -> {b.stride()}")
print(f"  contiguous? original {a.is_contiguous()}, transposed {b.is_contiguous()}")

print("\n=== so how fast is a 'free' transpose, really ===")
dev = "cuda:0"
x = torch.randn(8192, 8192, device=dev)
torch.cuda.synchronize()
t = time.perf_counter()
for _ in range(100): y = x.T
torch.cuda.synchronize()
print(f"  100 transposes of a 8192x8192 matrix: {(time.perf_counter()-t)*1e6:.0f} microseconds total")
t = time.perf_counter()
for _ in range(100): z = x.T.contiguous()
torch.cuda.synchronize()
print(f"  100 transposes THEN forced into a real line: {(time.perf_counter()-t)*1e3:.0f} ms")
print("  the second one actually moves 268 MB each time. The first moves nothing.")

print("\n=== the same numbers, four different shapes, one piece of memory ===")
flat = torch.arange(12, dtype=torch.float32)
for shp in [(12,), (3,4), (4,3), (2,2,3)]:
    v = flat.view(shp)
    print(f"  view{str(shp):<10} stride {str(v.stride()):<14} same memory: {v.data_ptr()==flat.data_ptr()}")
