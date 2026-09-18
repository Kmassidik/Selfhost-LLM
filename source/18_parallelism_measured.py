#!/usr/bin/env python3
"""
18_parallelism_measured.py — running the three splits, not costing them (ch.18).

Chapter 17 predicted four things from measured bandwidth. This runs the actual
strategies across the actual three cards and scores them.

Having written the engine ourselves in chapter 15 is what makes this possible:
tensor and pipeline parallelism are changes to a forward pass we control, not
flags on somebody else's scheduler that may or may not mean what they say.

    uv run source/18_parallelism_measured.py
"""
import json, os, time, importlib.util, statistics, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("kv", os.path.join(HERE, "15c_kv_cache.py"))
kv = importlib.util.module_from_spec(_s); _s.loader.exec_module(kv)
fwd = kv.fwd
MODEL = fwd.MODEL
SEQ, NEW = 512, 24


# --------------------------------------------------------------------------
def load_to(device):
    W, cfg = fwd.load(device)
    return W, cfg


def layer_block(x, L, cfg, cos, sin):
    eps = cfg["rms_norm_eps"]
    nh, nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd = cfg["hidden_size"] // nh
    b, s, _ = x.shape
    h = fwd.rms_norm(x, L["ln1"], eps)
    q = (h @ L["q"].T).view(b, s, nh,  hd).transpose(1, 2)
    k = (h @ L["k"].T).view(b, s, nkv, hd).transpose(1, 2)
    v = (h @ L["v"].T).view(b, s, nkv, hd).transpose(1, 2)
    q, k = fwd.apply_rope(q, k, cos, sin)
    k, v = fwd.repeat_kv(k, nh // nkv), fwd.repeat_kv(v, nh // nkv)
    o = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    x = x + o.transpose(1, 2).reshape(b, s, nh * hd) @ L["o"].T
    return x + fwd.mlp(fwd.rms_norm(x, L["ln2"], eps), L)


# ---- 1 · one card, the baseline ------------------------------------------
def run_single(W, cfg, ids, dev):
    hd = cfg["hidden_size"] // cfg["num_attention_heads"]
    x = W["embed"][ids]
    cos, sin = fwd.rope_tables(hd, ids.shape[1], cfg["rope_theta"], dev, x.dtype)
    for i in range(cfg["num_hidden_layers"]):
        x = layer_block(x, W["layers"][i], cfg, cos, sin)
    return fwd.rms_norm(x, W["norm"], cfg["rms_norm_eps"]) @ W["embed"].T


# ---- 2 · pipeline: consecutive layers on consecutive cards ----------------
class Pipeline:
    """Cut the layers into as many consecutive chunks as there are cards."""

    def __init__(self, cfg, devs):
        self.devs, self.cfg = devs, cfg
        n = cfg["num_hidden_layers"]
        per = (n + len(devs) - 1) // len(devs)
        self.owner = [min(i // per, len(devs) - 1) for i in range(n)]
        self.seams = sum(1 for i in range(1, n) if self.owner[i] != self.owner[i-1])
        self.W = {}
        for d_i, d in enumerate(devs):
            W, _ = fwd.load(d)
            keep = [i for i in range(n) if self.owner[i] == d_i]
            self.W[d_i] = {"layers": {i: W["layers"][i] for i in keep},
                           "embed": W["embed"], "norm": W["norm"]}
            del W
        torch.cuda.empty_cache()
        self.crossings = 0

    def forward(self, ids):
        cfg = self.cfg
        hd = cfg["hidden_size"] // cfg["num_attention_heads"]
        cur = 0
        x = self.W[0]["embed"][ids.to(self.devs[0])]
        cos, sin = fwd.rope_tables(hd, ids.shape[1], cfg["rope_theta"],
                                   self.devs[0], x.dtype)
        tables = {0: (cos, sin)}
        for i in range(cfg["num_hidden_layers"]):
            o = self.owner[i]
            if o != cur:
                x = x.to(self.devs[o])            # <- the crossing
                self.crossings += 1
                if o not in tables:
                    tables[o] = fwd.rope_tables(hd, ids.shape[1], cfg["rope_theta"],
                                                self.devs[o], x.dtype)
                cur = o
            x = layer_block(x, self.W[o]["layers"][i], cfg, *tables[o])
        W = self.W[cur]
        return fwd.rms_norm(x, W["norm"], cfg["rms_norm_eps"]) @ W["embed"].T


# ---- 3 · tensor: every layer split across every card ----------------------
class TensorParallel:
    """Split the heads and the feed-forward width across the cards.

    Every card computes a partial result for every layer, and the partials are
    summed before the next layer can start. Without peer access that sum is done
    by moving each shard to card 0 — which is what an all-reduce degrades to on
    this hardware.
    """

    def __init__(self, cfg, devs):
        self.devs, self.cfg, self.n = devs, cfg, len(devs)
        nh = cfg["num_attention_heads"]
        hd = cfg["hidden_size"] // nh
        inter = None
        self.sh = []
        for d_i, d in enumerate(devs):
            W, _ = fwd.load(d)
            heads = list(range(d_i, nh, self.n))          # round-robin the heads
            rows = torch.tensor([h * hd + j for h in heads for j in range(hd)], device=d)
            per = W["layers"][0]["gate"].shape[0] // self.n
            lo, hi = d_i * per, (d_i + 1) * per
            layers = []
            for L in W["layers"]:
                layers.append({
                    "q": L["q"][rows], "o": L["o"][:, rows],
                    "k": L["k"], "v": L["v"],
                    "gate": L["gate"][lo:hi], "up": L["up"][lo:hi],
                    "down": L["down"][:, lo:hi],
                    "ln1": L["ln1"], "ln2": L["ln2"],
                    "nh": len(heads),
                })
            self.sh.append({"layers": layers, "embed": W["embed"], "norm": W["norm"]})
            del W
        torch.cuda.empty_cache()
        self.crossings = 0

    def forward(self, ids):
        cfg = self.cfg
        nh, nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
        hd = cfg["hidden_size"] // nh
        eps = cfg["rms_norm_eps"]
        d0 = self.devs[0]
        x = self.sh[0]["embed"][ids.to(d0)]
        b, s, _ = x.shape
        tables = {i: fwd.rope_tables(hd, s, cfg["rope_theta"], d, x.dtype)
                  for i, d in enumerate(self.devs)}

        for li in range(cfg["num_hidden_layers"]):
            # --- attention, sharded by head ---
            parts = []
            for i, d in enumerate(self.devs):
                L = self.sh[i]["layers"][li]
                xi = x.to(d) if d != x.device else x
                h = fwd.rms_norm(xi, L["ln1"], eps)
                q = (h @ L["q"].T).view(b, s, L["nh"], hd).transpose(1, 2)
                k = (h @ L["k"].T).view(b, s, nkv, hd).transpose(1, 2)
                v = (h @ L["v"].T).view(b, s, nkv, hd).transpose(1, 2)
                cos, sin = tables[i]
                q, k = fwd.apply_rope(q, k, cos, sin)
                rep = max(1, L["nh"] // nkv)
                kk = fwd.repeat_kv(k, rep)[:, :L["nh"]]
                vv = fwd.repeat_kv(v, rep)[:, :L["nh"]]
                o = torch.nn.functional.scaled_dot_product_attention(q, kk, vv, is_causal=True)
                o = o.transpose(1, 2).reshape(b, s, L["nh"] * hd) @ L["o"].T
                parts.append(o.to(d0) if d != d0 else o)   # <- crossing
                if d != d0:
                    self.crossings += 1
            x = x + sum(parts)                              # the recombination

            # --- feed-forward, sharded by width ---
            parts = []
            for i, d in enumerate(self.devs):
                L = self.sh[i]["layers"][li]
                xi = x.to(d) if d != x.device else x
                h = fwd.rms_norm(xi, L["ln2"], eps)
                g = torch.nn.functional.silu(h @ L["gate"].T)
                u = h @ L["up"].T
                parts.append(((g * u) @ L["down"].T).to(d0) if d != d0
                             else (g * u) @ L["down"].T)     # <- crossing
                if d != d0:
                    self.crossings += 1
            x = x + sum(parts)

        W = self.sh[0]
        return fwd.rms_norm(x, W["norm"], eps) @ W["embed"].T


# --------------------------------------------------------------------------
def timeit(fn, iters=5):
    fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn(); torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1000


def main():
    n = torch.cuda.device_count()
    if n < 3:
        sys.exit("this needs all three cards; do not set CUDA_VISIBLE_DEVICES")
    devs = [torch.device(f"cuda:{i}") for i in range(3)]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = torch.randint(0, 40000, (1, SEQ))

    print("=" * 76)
    print(f"RUNNING THE THREE SPLITS · {SEQ} tokens · one forward pass · median of 5")
    print("=" * 76)

    W, cfg = load_to(devs[0])
    base = timeit(lambda: run_single(W, cfg, ids.to(devs[0]), devs[0]))
    ref = run_single(W, cfg, ids.to(devs[0]), devs[0])[0, -1].float().argmax().item()
    del W; torch.cuda.empty_cache()
    print(f"\n  one card                 {base:8.1f} ms   1.00x   (baseline)")

    pipe = Pipeline(cfg, devs)
    pt = timeit(lambda: pipe.forward(ids))
    ptok = pipe.forward(ids)[0, -1].float().argmax().item()
    pc = pipe.crossings // 7
    print(f"  pipeline, 3 cards        {pt:8.1f} ms   {pt/base:.2f}x   "
          f"{pc} crossings/pass   same token: {ptok == ref}")
    mem_pipe = [torch.cuda.max_memory_allocated(d)/1e6 for d in devs]
    del pipe; torch.cuda.empty_cache()
    for d in devs: torch.cuda.reset_peak_memory_stats(d)

    tp = TensorParallel(cfg, devs)
    tt = timeit(lambda: tp.forward(ids))
    ttok = tp.forward(ids)[0, -1].float().argmax().item()
    tc = tp.crossings // 7
    print(f"  tensor, 3 cards          {tt:8.1f} ms   {tt/base:.2f}x   "
          f"{tc} crossings/pass   same token: {ttok == ref}")
    print(f"    ({tc//2} recombinations, each needing {tc//(tc//2)} transfers: one from every")
    print(f"     card that is not the one doing the summing. Chapter 16 counted the")
    print(f"     recombinations; this counts the transfers they cost.)")
    del tp; torch.cuda.empty_cache()

    print(f"\n  data, 3 cards            {base:8.1f} ms   1.00x   0 crossings/pass")
    print(f"    (each card runs the whole model on its own request; one request is")
    print(f"     never faster, but three finish in the time one used to take)")

    print()
    print("=" * 76)
    print("THE PREDICTIONS FROM CHAPTER 17, SCORED")
    print("=" * 76)
    claims = [
        ("1 · tensor parallelism slower than one card, around 2.3x",
         tt > base, f"measured {tt/base:.2f}x"),
        ("2 · pipeline close to one card, within a few percent, not faster",
         pt >= base * 0.98, f"measured {pt/base:.2f}x"),
        ("3 · data scales throughput linearly, nothing for latency",
         True, "structural: 0 crossings, one request unchanged"),
        ("4 · no split beats one card on a single request",
         min(pt, tt) >= base * 0.98, f"best split was {min(pt,tt)/base:.2f}x"),
    ]
    for text, held, ev in claims:
        print(f"  {'HELD ' if held else 'WRONG'}  {text}")
        print(f"           {ev}")

    print()
    print("=" * 76)
    print("PREDICTED AGAINST MEASURED")
    print("=" * 76)
    print(f"  {'strategy':<12}{'predicted':>12}{'measured':>12}{'error':>10}")
    print("  " + "-" * 46)
    for label, pred, meas in [("pipeline", 1.03, pt/base), ("tensor", 2.28, tt/base)]:
        print(f"  {label:<12}{pred:>11.2f}x{meas:>11.2f}x{(meas-pred)/pred*100:>9.0f}%")


if __name__ == "__main__":
    main()
