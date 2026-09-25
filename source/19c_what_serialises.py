"""The cards overlap. The pipeline did not. What is serialising it?

Prime suspect: the handoff. Chapter 17 established these cards have no peer
access, so a card-to-card copy is staged through host memory — and a staged
copy has to complete before the receiving card can use the data.

Test: run the same wavefront twice. Once with the real handoff, and once with
the transfer removed (each stage works on a tensor already sitting on its card).
If the second one fills and the first does not, the handoff is the serialiser.
"""
import os, time, statistics, importlib.util, torch
HERE = "/root/Desktop/selfhostllm/source"
_s = importlib.util.spec_from_file_location("par", HERE + "/18_parallelism_measured.py")
par = importlib.util.module_from_spec(_s); _s.loader.exec_module(par)
fwd = par.fwd

devs = [torch.device(f"cuda:{i}") for i in range(3)]
SEQ = 512
W, cfg = fwd.load(devs[0])
hd = cfg["hidden_size"] // cfg["num_attention_heads"]
del W; torch.cuda.empty_cache()

# one stage's worth of layers, resident on each card
Ws, tabs = [], []
for d in devs:
    w, _ = fwd.load(d)
    Ws.append([w["layers"][i] for i in range(11)])
    tabs.append(fwd.rope_tables(hd, SEQ, cfg["rope_theta"], d, torch.bfloat16))
    del w
torch.cuda.empty_cache()
resident = [torch.randn(1, SEQ, cfg["hidden_size"], dtype=torch.bfloat16, device=d) for d in devs]
streams = [torch.cuda.Stream(device=d) for d in devs]

def stage(s, x):
    for L in Ws[s]:
        x = par.layer_block(x, L, cfg, *tabs[s])
    return x

def med(fn, iters=3):
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

def wave(R, handoff):
    held = {}
    for t in range(R + 2):
        for s in (2, 1, 0):
            r = t - s
            if not (0 <= r < R): continue
            with torch.cuda.stream(streams[s]):
                if s == 0:
                    held[r] = stage(0, resident[0])
                elif handoff:
                    held[r] = stage(s, held[r].to(devs[s], non_blocking=True))
                else:
                    held[r] = stage(s, resident[s])     # no transfer at all

print(f"{'requests':>9}{'with handoff':>16}{'no handoff':>14}{'handoff cost':>15}")
print("-" * 55)
base_h = base_n = None
for R in (1, 3, 6):
    h = med(lambda R=R: wave(R, True))
    n = med(lambda R=R: wave(R, False))
    if base_h is None: base_h, base_n = h, n
    print(f"{R:>9}{h:>13.1f}ms{n:>12.1f}ms{h-n:>13.1f}ms")
print()
h6 = med(lambda: wave(6, True)); h1 = med(lambda: wave(1, True))
n6 = med(lambda: wave(6, False)); n1 = med(lambda: wave(1, False))
print(f"6 requests vs 1 request:")
print(f"  with handoff : {h6/h1:.2f}x   (6.00x would mean no overlap at all)")
print(f"  no handoff   : {n6/n1:.2f}x")
print()
if n6/n1 < h6/h1 * 0.75:
    print("Removing the transfer lets the pipeline overlap. The handoff is the")
    print("serialiser — and chapter 17 already explained why it is expensive:")
    print("no peer access, so every crossing is staged through host memory and")
    print("must complete before the receiving stage can begin.")
else:
    print("Removing the transfer changes little, so the handoff is not the")
    print("serialiser and something else in the loop is.")
