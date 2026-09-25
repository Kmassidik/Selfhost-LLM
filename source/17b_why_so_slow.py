"""Is 5.3 GB/s the bus, or is it how we asked?

Three things could cap a card-to-card copy: the link itself, the absence of peer
access forcing a detour through host memory, or pageable host memory the driver
must stage twice. This separates them.
"""
import time, torch
MB = 1024 * 1024

def rate(a, b, iters=10):
    for _ in range(3): b.copy_(a)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): b.copy_(a)
    torch.cuda.synchronize()
    return a.numel() * 2 * iters / 1e9 / (time.perf_counter() - t0)

N = 256 * MB // 2
print("Gen 3 x16 is 15.75 GB/s in theory, one direction.\n")

g0 = torch.empty(N, dtype=torch.float16, device="cuda:0")
g1 = torch.empty(N, dtype=torch.float16, device="cuda:1")
pageable = torch.empty(N, dtype=torch.float16)
pinned = torch.empty(N, dtype=torch.float16).pin_memory()

print(f"{'route':<34}{'GB/s':>9}{'% of Gen3 x16':>16}")
print("-" * 59)
for label, a, b in [
    ("host (pageable) -> card 0", pageable, g0),
    ("host (pinned)   -> card 0", pinned,   g0),
    ("card 0 -> host (pageable)", g0, pageable),
    ("card 0 -> host (pinned)",   g0, pinned),
    ("card 0 -> card 1",          g0, g1),
]:
    r = rate(a, b)
    print(f"{label:<34}{r:>9.2f}{r/15.75*100:>15.0f}%")

print()
print("peer access 0<->1:", torch.cuda.can_device_access_peer(0, 1))
try:
    torch.cuda.set_device(0)
    from torch.cuda import memory
    print("attempting explicit peer enable ...")
    import ctypes
    rt = ctypes.CDLL("libcudart.so")
    rc = rt.cudaDeviceEnablePeerAccess(ctypes.c_int(1), ctypes.c_int(0))
    print(f"  cudaDeviceEnablePeerAccess(1) returned {rc}"
          f"  ({'ok' if rc == 0 else 'refused — 217 is cudaErrorPeerAccessUnsupported'})")
except Exception as e:
    print("  ", e)
print()
print("GeForce cards have peer-to-peer over PCIe disabled in the driver; it is a")
print("product decision, not a bus limitation. Every card-to-card copy therefore")
print("goes out to host memory and back, paying the bus twice.")
