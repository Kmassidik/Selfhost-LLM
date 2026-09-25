#!/usr/bin/env python3
"""
16_parallelism_cost.py — what each way of splitting a model costs (ch.16).

Three ways to use more than one card. The textbook describes them as equally
valid choices. On a machine with a slow link they are not remotely equal, and
the difference is arithmetic rather than opinion.

Every input here is measured on this box:
  card-to-card    5.31 GB/s     17_interconnect.py, 2026-09-14
  fixed latency   43.8 us       17_interconnect.py, 2026-09-14
  per-layer compute 1.61 ms     11_ngl_sweep.py, chapter 11

    uv run source/16_parallelism_cost.py
"""
BW = 5.31e9          # bytes/s, best measured card-to-card route
LAT = 43.8e-6        # seconds, fixed cost of any transfer
LAYER_MS = 1.61      # ms of compute per layer, measured
CARDS = 3


def transfer(nbytes):
    return nbytes / BW + LAT


def report(name, layers, hidden, seq, dtype_bytes=2):
    act = seq * hidden * dtype_bytes                 # one layer's activations
    compute = layers * LAYER_MS / 1000.0             # seconds

    # DATA: every card holds the whole model and answers a different request.
    # Nothing crosses the link during inference at all.
    data_moves, data_bytes = 0, 0

    # PIPELINE: the model is cut into CARDS consecutive chunks. One handoff at
    # each seam, so CARDS-1 per forward pass.
    pipe_moves = CARDS - 1
    pipe_bytes = act

    # TENSOR: every layer is split across every card, so the pieces must be
    # recombined twice per layer — once after attention, once after the
    # feed-forward block. A ring all-reduce moves 2(N-1)/N of the tensor.
    tensor_moves = layers * 2
    tensor_bytes = act * 2 * (CARDS - 1) / CARDS

    print(f"\n{name}  —  {layers} layers, hidden {hidden}, {seq:,} tokens")
    print(f"  one layer's activations: {act/1e6:.2f} MB")
    print(f"  compute for a full pass: {compute*1000:.1f} ms")
    print()
    print(f"  {'strategy':<12}{'transfers':>11}{'bytes moved':>14}{'link time':>12}"
          f"{'total':>11}{'vs 1 card':>11}")
    print("  " + "-" * 71)
    rows = [("data", data_moves, data_bytes), ("pipeline", pipe_moves, pipe_bytes),
            ("tensor", tensor_moves, tensor_bytes)]
    for label, moves, per in rows:
        link = moves * transfer(per) if moves else 0.0
        # data parallelism does not make ONE request faster; it runs three at once
        total = compute + link
        print(f"  {label:<12}{moves:>11,}{moves*per/1e6:>13.1f} MB"
              f"{link*1000:>11.1f} ms{total*1000:>10.1f} ms{total/compute:>10.2f}x")
    print()
    print(f"  Memory per card: data needs the WHOLE model on each; pipeline and")
    print(f"  tensor each need about 1/{CARDS}.")


print("=" * 78)
print("WHAT EACH WAY OF SPLITTING COSTS — derived from measured numbers")
print("=" * 78)
print(f"  card-to-card {BW/1e9:.2f} GB/s · latency {LAT*1e6:.1f} us · "
      f"{LAYER_MS} ms compute per layer · {CARDS} cards")

report("SmolLM2-360M", 32, 960, 2048)
report("an 8B-shaped model", 32, 4096, 2048)

print()
print("=" * 78)
print("THE POINT")
print("=" * 78)
print("""
  Tensor parallelism pays the link twice per layer. With 32 layers that is 64
  crossings for a single forward pass, and on this machine each one costs more
  than the fixed latency of the compute it is interrupting.

  Pipeline parallelism pays it twice TOTAL, because the model has only two
  internal seams when it is cut in three.

  Data parallelism never pays it at all, and cannot help a model that does not
  fit on one card — which is the only reason anyone reaches for the other two.

  On NVLink at 600 GB/s the tensor row would cost about 113x less link time and
  the choice would look entirely different. That is why the standard advice
  does not transfer: it was measured on a machine where the link is nearly free.
""")
