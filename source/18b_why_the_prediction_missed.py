"""Why was the tensor prediction 39% optimistic?

Chapter 17 costed tensor parallelism using the textbook all-reduce: a RING,
where each card sends 2(N-1)/N of the tensor to its neighbour and the sum
assembles as it goes. A ring needs every card to send to every other card
directly. These cards refuse peer access, so there is no ring — every shard is
carried to one card, which sums them. That moves more data.
"""
BW, LAT = 5.31e9, 43.8e-6
LAYERS, HIDDEN, SEQ, N = 32, 960, 512, 3
COMPUTE_MS = 61.3          # measured this run, one card, 512 tokens

act = SEQ * HIDDEN * 2     # one layer's activations, bytes
print(f"one layer's activations at {SEQ} tokens: {act/1e6:.2f} MB\n")

def cost(label, transfers, per_transfer_bytes):
    t = transfers * (per_transfer_bytes / BW + LAT)
    total = COMPUTE_MS / 1000 + t
    print(f"  {label:<34}{transfers:>6} transfers{t*1000:>9.1f} ms link"
          f"{total*1000:>9.1f} ms{total/(COMPUTE_MS/1000):>8.2f}x")
    return total / (COMPUTE_MS / 1000)

print(f"  {'model of the all-reduce':<34}{'':>6}{'':>9}{'':>18}{'total':>8}")
print("  " + "-" * 78)
ring = cost("ring (what ch.17 assumed)", LAYERS * 2, act * 2 * (N - 1) / N)
gather = cost("gather to one card (what runs)", LAYERS * 2 * (N - 1), act)
print()
print(f"  measured                                                            3.18x")
print()
print(f"  ring model       {ring:.2f}x  -> off by {(3.18-ring)/3.18*100:.0f}%")
print(f"  gather model     {gather:.2f}x  -> off by {(3.18-gather)/3.18*100:+.0f}%")
print(f"""
  The ring model under-predicts because a ring is not available here. It needs
  each card to hand data straight to the next one, which is exactly what peer
  access provides and exactly what these cards refuse. The fallback carries
  every shard to a single card: {N-1} full tensors per recombination instead of
  {2*(N-1)/N:.2f}, which is {(N-1)/(2*(N-1)/N):.1f}x the traffic.

  So the standard cost model for tensor parallelism quietly assumes peer
  access. On hardware without it, the textbook formula is optimistic before
  the bandwidth number is even plugged in.
""")
