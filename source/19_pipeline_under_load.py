#!/usr/bin/env python3
"""
19_pipeline_under_load.py — three cards, one sequence (ch.19).

Chapter 18 measured pipeline parallelism on a single forward pass and found it
costs 1.02x. That number hides the thing that matters for serving.

A pipeline is stations in a row. While station one works, stations two and three
have nothing to do — the work has not reached them yet. One request travelling
through three stages uses one card at a time, so three cards do one card's work
and two thirds of the hardware idles. The single-pass timing cannot show this,
because it measures the journey rather than the occupancy.

This measures the occupancy, then measures what fixes it.

    uv run source/19_pipeline_under_load.py
"""
import os, time, statistics, importlib.util, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("par", os.path.join(HERE, "18_parallelism_measured.py"))
par = importlib.util.module_from_spec(_s); _s.loader.exec_module(par)
fwd, kv = par.fwd, par.kv
SEQ = 512


def med(fn, iters=5):
    fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn(); torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


class Stages:
    """The pipeline from chapter 18, but able to run one stage at a time."""

    def __init__(self, cfg, devs):
        self.cfg, self.devs = cfg, devs
        n = cfg["num_hidden_layers"]
        per = (n + len(devs) - 1) // len(devs)
        self.owner = [min(i // per, len(devs) - 1) for i in range(n)]
        self.mine = [[i for i in range(n) if self.owner[i] == d] for d in range(len(devs))]
        self.W = {}
        for d_i, d in enumerate(devs):
            W, _ = fwd.load(d)
            self.W[d_i] = {"layers": {i: W["layers"][i] for i in self.mine[d_i]},
                           "embed": W["embed"], "norm": W["norm"]}
            del W
        torch.cuda.empty_cache()
        hd = cfg["hidden_size"] // cfg["num_attention_heads"]
        self.tab = {i: fwd.rope_tables(hd, SEQ, cfg["rope_theta"], d, torch.bfloat16)
                    for i, d in enumerate(devs)}

    def stage(self, s, x):
        """Run only the layers that live on card s."""
        for i in self.mine[s]:
            x = par.layer_block(x, self.W[s]["layers"][i], self.cfg, *self.tab[s])
        return x

    def embed(self, ids):
        return self.W[0]["embed"][ids.to(self.devs[0])]

    def head(self, x):
        W = self.W[len(self.devs) - 1]
        return fwd.rms_norm(x, W["norm"], self.cfg["rms_norm_eps"]) @ W["embed"].T


def main():
    if torch.cuda.device_count() < 3:
        sys.exit("needs all three cards")
    devs = [torch.device(f"cuda:{i}") for i in range(3)]
    W0, cfg = fwd.load(devs[0])
    ids = torch.randint(0, 40000, (1, SEQ))
    single = med(lambda: par.run_single(W0, cfg, ids.to(devs[0]), devs[0]))
    del W0; torch.cuda.empty_cache()

    st = Stages(cfg, devs)

    # ---- 1 · how long is each stage busy? ---------------------------------
    print("=" * 74)
    print("1 · WHO IS WORKING, AND FOR HOW LONG")
    print("=" * 74)
    x0 = st.embed(ids)
    times = []
    xs = x0
    for s in range(3):
        xin = xs if s == 0 else xs.to(devs[s])
        t = med(lambda s=s, xin=xin: st.stage(s, xin))
        times.append(t)
        xs = st.stage(s, xin)
    tot = sum(times)
    print(f"  {'stage':<10}{'layers':>8}{'busy':>11}{'share of the pass':>20}")
    print("  " + "-" * 50)
    for s in range(3):
        print(f"  card {s:<5}{len(st.mine[s]):>8}{times[s]*1000:>10.1f}ms{times[s]/tot*100:>19.0f}%")
    print(f"  {'total':<10}{'32':>8}{tot*1000:>10.1f}ms")
    print()
    print(f"  While any one stage runs, the other two are idle. For a single")
    print(f"  request this is not a detail — it is {2/3*100:.0f}% of the hardware,")
    print(f"  waiting, for the whole pass.")

    # ---- 2 · one sequence, token by token ---------------------------------
    print()
    print("=" * 74)
    print("2 · ONE SEQUENCE THROUGH THE PIPELINE")
    print("=" * 74)
    def one_pass():
        x = st.embed(ids)
        for s in range(3):
            if s: x = x.to(devs[s])
            x = st.stage(s, x)
        return st.head(x)
    pipe = med(one_pass)
    print(f"  one card                {single*1000:>8.1f} ms   card busy {100:.0f}% of the time")
    print(f"  pipeline, one request   {pipe*1000:>8.1f} ms   each card busy "
          f"{tot/3/pipe*100:.0f}% of the time")
    print()
    print(f"  Three cards. {tot/3/pipe*100:.0f}% utilisation each. The pass takes as long as")
    print(f"  it did on one card because the work is the same work, done in the")
    print(f"  same order, merely spread over hardware that takes turns.")

    # ---- 3 · fill the pipeline --------------------------------------------
    print()
    print("=" * 74)
    print("3 · FILLING IT — several requests in flight at once")
    print("=" * 74)
    print("  While card 0 starts request N, card 1 can be working on N-1 and card")
    print("  2 on N-2. Nothing waits, provided there is always another request.")
    print()
    print(f"  {'requests':>9}{'wall time':>12}{'per request':>14}{'throughput':>13}{'vs 1 card':>11}")
    print("  " + "-" * 60)
    streams = [torch.cuda.Stream(device=d) for d in devs]
    for R in (1, 2, 3, 6):
        def wave(R=R):
            # stage s of request r can start once stage s-1 of r has finished.
            # Run it as a wavefront: at step t, card s handles request t-s.
            held = {}
            steps = R + 2
            for t in range(steps):
                for s in (2, 1, 0):
                    r = t - s
                    if not (0 <= r < R):
                        continue
                    with torch.cuda.stream(streams[s]):
                        if s == 0:
                            held[r] = st.stage(0, st.embed(ids))
                        else:
                            held[r] = st.stage(s, held[r].to(devs[s], non_blocking=True))
            for d in devs:
                torch.cuda.synchronize(d)
        w = med(wave, iters=3)
        print(f"  {R:>9}{w*1000:>11.1f}ms{w/R*1000:>13.1f}ms"
              f"{R/w:>12.1f}/s{(R/w)/(1/single):>10.2f}x")

    print()
    print("=" * 74)
    print("WHAT THIS MEANS FOR SERVING")
    print("=" * 74)
    print(f"""
  Pipeline parallelism does not make a request faster. It never could — the
  stages are sequential and a single request visits them one at a time.

  What it buys is that a model too large for one card RUNS, at close to
  single-card speed, and that the idle stages fill up as soon as there is more
  than one request to work on.

  The 1.02x from chapter 18 was the journey time for one request. It was true,
  and it was not the number a server cares about.
""")


if __name__ == "__main__":
    main()
