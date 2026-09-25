#!/usr/bin/env python3
"""
17_interconnect.py — the number Part II rests on (ch.17).

Part I never moved a byte between cards. Part II cannot avoid it, so the first
thing to establish is what that costs on THIS machine — not on the hardware the
tutorials were written for.

Chapter 03 measured reading from a card's own memory at 423.9 GB/s. Chapter 06
read the topology and found card 0 on a different root complex from cards 1 and
2. This measures the thing those two facts bracket: how fast one card can hand
data to another, and how that compares both to its own memory and to the NVLink
number quoted in every multi-GPU guide.

    uv run source/17_interconnect.py
"""
import time, sys
import torch

MB = 1024 * 1024


def link_state(tag):
    import subprocess
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,pcie.link.gen.current,pcie.link.gen.max,"
         "pcie.link.width.current,pcie.link.width.max", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip()
    print(f"  PCIe link {tag}:")
    for line in out.split("\n"):
        i, g, gm, w, wm = [x.strip() for x in line.split(",")]
        # Gen 3 x16 is ~15.75 GB/s one way; each generation roughly doubles per lane
        per_lane = {1: 0.25, 2: 0.5, 3: 0.985, 4: 1.969, 5: 3.938}
        cur = per_lane.get(int(g), 0) * int(w)
        mx = per_lane.get(int(gm), 0) * int(wm)
        print(f"    card {i}: gen {g} x{w}  (max gen {gm} x{wm})"
              f"   theoretical {cur:.1f} GB/s now, {mx:.1f} GB/s at full rate")


def bandwidth(src, dst, size_mb=256, iters=10):
    """Sustained copy rate between two devices, steady state."""
    a = torch.empty(size_mb * MB // 2, dtype=torch.float16, device=src)
    b = torch.empty_like(a, device=dst)
    for _ in range(3):                       # warm up, and let the link clock up
        b.copy_(a)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        b.copy_(a)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return size_mb * iters / 1024 / dt        # GB/s


def latency(src, dst, iters=200):
    """Cost of a tiny transfer — what a per-layer handoff actually pays."""
    a = torch.empty(4, dtype=torch.float16, device=src)
    b = torch.empty_like(a, device=dst)
    for _ in range(20):
        b.copy_(a)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        b.copy_(a)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6      # microseconds


def main():
    n = torch.cuda.device_count()
    print(f"cards visible: {n}")
    if n < 2:
        sys.exit("need at least two cards — do not set CUDA_VISIBLE_DEVICES for this one")
    for i in range(n):
        print(f"  card {i}: {torch.cuda.get_device_name(i)}")
    print()
    link_state("at rest")

    # ---- peer to peer: can card A read card B's memory directly? -----------
    print()
    print("=" * 74)
    print("1 · PEER ACCESS — can one card address another's memory directly?")
    print("=" * 74)
    for i in range(n):
        row = []
        for j in range(n):
            row.append("self" if i == j else
                       ("yes" if torch.cuda.can_device_access_peer(i, j) else "NO"))
        print(f"  card {i} -> {row}")
    print("  Without peer access every card-to-card copy is staged through host memory:")
    print("  two transfers over the same bus instead of one.")

    # ---- sustained bandwidth, every ordered pair ---------------------------
    print()
    print("=" * 74)
    print("2 · SUSTAINED BANDWIDTH — 256 MB copies, steady state")
    print("=" * 74)
    print(f"  {'route':<22}{'GB/s':>10}   vs one card's own memory (423.9 GB/s)")
    print("  " + "-" * 66)
    results = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            bw = bandwidth(f"cuda:{i}", f"cuda:{j}")
            results[(i, j)] = bw
            print(f"  card {i} -> card {j}{'':<8}{bw:>10.2f}   {423.9/bw:>5.1f}x slower")
    hb = bandwidth("cpu", "cuda:0")
    hb2 = bandwidth("cuda:0", "cpu")
    print(f"  host -> card 0{'':<8}{hb:>10.2f}   {423.9/hb:>5.1f}x slower")
    print(f"  card 0 -> host{'':<8}{hb2:>10.2f}   {423.9/hb2:>5.1f}x slower")

    print()
    link_state("under load")

    # ---- latency ----------------------------------------------------------
    print()
    print("=" * 74)
    print("3 · LATENCY — the fixed cost of any handoff, however small")
    print("=" * 74)
    print(f"  {'route':<22}{'microseconds':>14}")
    print("  " + "-" * 38)
    for (i, j) in list(results)[:3]:
        print(f"  card {i} -> card {j}{'':<8}{latency(f'cuda:{i}', f'cuda:{j}'):>14.1f}")

    # ---- what it means ----------------------------------------------------
    print()
    print("=" * 74)
    print("4 · WHAT THIS COSTS A SPLIT MODEL")
    print("=" * 74)
    best = max(results.values())
    worst = min(results.values())
    print(f"  best card-to-card route  : {best:.2f} GB/s")
    print(f"  worst card-to-card route : {worst:.2f} GB/s")
    print(f"  one card's own memory    : 423.9 GB/s   (measured, chapter 03)")
    print(f"  NVLink, quoted           : 600 GB/s     (not this machine)")
    print()
    print(f"  Reading from a card's own memory is {423.9/best:.0f}x faster than the best")
    print(f"  route between two of them, and {600/best:.0f}x is the gap to the hardware the")
    print("  standard multi-GPU advice was written on.")
    print()
    # a 360M model's per-layer activation, 2048 tokens, bf16
    act = 2048 * 960 * 2 / 1e9
    print(f"  Concretely: one layer's activations for 2,048 tokens of SmolLM2-360M")
    print(f"  is {act*1000:.1f} MB. Handing that between cards costs {act/best*1e6:.0f} us of")
    print(f"  transfer plus ~{latency('cuda:0','cuda:1'):.0f} us of fixed latency, against the")
    print(f"  ~1.61 ms per layer chapter 11 measured for the compute itself.")


if __name__ == "__main__":
    main()
