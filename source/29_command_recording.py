#!/usr/bin/env python3
"""
29_command_recording.py — the fix three chapters named and none of them built.

Chapter 05 measured naive PyTorch at 16.1 tok/s against a 584 ceiling and called
the 36x gap launch overhead rather than algorithm. Chapter 13 measured SGLang
falling from 258.6 to 24.0 tok/s per stream when command recording was switched
off. Chapter 15 built an engine that landed at 16.3. Chapter 19 found three
cards refusing to overlap because one thread was issuing kernels one at a time.

Four symptoms, one named cause, never addressed. This addresses it.

WHAT A CUDA GRAPH IS. Every operation normally costs a launch: the CPU tells the
card what to do, and that telling has a fixed price whatever the work. A graph
records the whole sequence ONCE and replays it with a single instruction, so 
launching a decode step costs one launch instead of hundreds.

WHAT IT COSTS. The recording is fixed. Every tensor must live at the same address
on every replay, so the cache is pre-allocated to its maximum and written in
place, and the sequence length can no longer change the shape of anything.

    uv run source/29_command_recording.py
"""
import json, os, statistics, sys, time, importlib.util
import torch
import torch.nn.functional as F

ROOT = "/root/Desktop/selfhostllm"
_s = importlib.util.spec_from_file_location("kv", f"{ROOT}/source/15c_kv_cache.py")
kv = importlib.util.module_from_spec(_s); _s.loader.exec_module(kv)
fwd = kv.fwd
MAXLEN = 512


class StaticEngine:
    """The chapter-15 forward pass, with everything pinned in place.

    The cache is allocated once at full size. A step writes into row `pos` and
    reads rows 0..pos, masking the rest — so no tensor ever changes shape and
    the same recorded sequence is valid at every position.
    """

    def __init__(self, W, cfg, dev):
        self.W, self.cfg, self.dev = W, cfg, dev
        L = cfg["num_hidden_layers"]
        nkv, hd = cfg["num_key_value_heads"], cfg["hidden_size"] // cfg["num_attention_heads"]
        self.k = [torch.zeros(1, nkv, MAXLEN, hd, dtype=torch.bfloat16, device=dev) for _ in range(L)]
        self.v = [torch.zeros(1, nkv, MAXLEN, hd, dtype=torch.bfloat16, device=dev) for _ in range(L)]
        full_cos, full_sin = fwd.rope_tables(hd, MAXLEN, cfg["rope_theta"], dev, torch.bfloat16)
        self.cos_all, self.sin_all = full_cos, full_sin
        # the three things a step needs, at fixed addresses
        self.tok = torch.zeros(1, 1, dtype=torch.long, device=dev)
        self.pos = torch.zeros(1, dtype=torch.long, device=dev)
        self.mask = torch.zeros(1, 1, 1, MAXLEN, dtype=torch.bfloat16, device=dev)
        self.out = None
        self.graph = None

    def _step(self):
        cfg, W = self.cfg, self.W
        nh, nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
        hd = cfg["hidden_size"] // nh
        eps = cfg["rms_norm_eps"]
        x = W["embed"][self.tok]
        cos = self.cos_all.index_select(0, self.pos).view(1, 1, 1, hd)
        sin = self.sin_all.index_select(0, self.pos).view(1, 1, 1, hd)
        for i in range(cfg["num_hidden_layers"]):
            L = W["layers"][i]
            h = fwd.rms_norm(x, L["ln1"], eps)
            q = (h @ L["q"].T).view(1, 1, nh,  hd).transpose(1, 2)
            k = (h @ L["k"].T).view(1, 1, nkv, hd).transpose(1, 2)
            v = (h @ L["v"].T).view(1, 1, nkv, hd).transpose(1, 2)
            q = q * cos + fwd.rotate_half(q) * sin
            k = k * cos + fwd.rotate_half(k) * sin
            # write in place at `pos` — the address never moves
            self.k[i].index_copy_(2, self.pos, k)
            self.v[i].index_copy_(2, self.pos, v)
            kk = fwd.repeat_kv(self.k[i], nh // nkv)
            vv = fwd.repeat_kv(self.v[i], nh // nkv)
            o = F.scaled_dot_product_attention(q, kk, vv, attn_mask=self.mask)
            x = x + o.transpose(1, 2).reshape(1, 1, nh * hd) @ L["o"].T
            x = x + fwd.mlp(fwd.rms_norm(x, L["ln2"], eps), L)
        return fwd.rms_norm(x, W["norm"], eps) @ W["embed"].T

    def set_pos(self, p):
        self.pos.fill_(p)
        m = torch.full((MAXLEN,), float("-inf"), dtype=torch.bfloat16, device=self.dev)
        m[:p + 1] = 0
        self.mask.copy_(m.view(1, 1, 1, MAXLEN))

    def capture(self):
        """Record one decode step. Warm up on a side stream first, as required."""
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._step()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = self._step()
        torch.cuda.synchronize()

    def replay(self):
        self.graph.replay()
        return self.out


def bench(fn, n=64):
    for _ in range(8):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return n / (time.perf_counter() - t0)


def main():
    dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
    W, cfg = fwd.load(dev)
    eng = StaticEngine(W, cfg, dev)
    eng.tok.fill_(504)
    eng.set_pos(16)

    print("=" * 74)
    print("COMMAND RECORDING — the fix chapters 05, 13, 15 and 19 all named")
    print("=" * 74)
    print(f"  model            SmolLM2-360M, {cfg['num_hidden_layers']} layers")
    print(f"  measuring        one decode step, batch 1, position 16\n")

    eager = bench(eng._step)
    print(f"  {'issuing every kernel (what ch.15 built)':<44}{eager:>8.1f} tok/s")

    eng.capture()
    graphed = bench(eng.replay)
    print(f"  {'replaying one recorded graph':<44}{graphed:>8.1f} tok/s")
    print(f"  {'':<44}{'':>8}")
    print(f"  {'speedup':<44}{graphed/eager:>8.2f}x")

    # does it still produce the same token?
    eng.set_pos(16); eng.tok.fill_(504)
    a = int(eng._step()[0, -1].argmax())
    eng.replay()
    b = int(eng.out[0, -1].argmax())
    print(f"\n  same token from both paths: {a == b}   (eager {a}, graphed {b})")

    print()
    print("=" * 74)
    print("AGAINST EVERY OTHER NUMBER IN THIS GUIDE")
    print("=" * 74)
    rows = [("ch.05 naive PyTorch", 16.1), ("ch.15 our engine, as built", 16.3),
            ("ch.13 SGLang, recording OFF", 24.0), ("this, recorded", graphed),
            ("ch.14 Ollama", 290.1), ("ch.13 SGLang, recording ON", 258.6),
            ("ch.03 memory-bandwidth ceiling", 584.0)]
    for label, v in sorted(rows, key=lambda r: r[1]):
        bar = "█" * max(1, int(v / 584 * 46))
        print(f"  {label:<32}{v:>7.1f}  {bar}")
    print(f"\n  ceiling reached: {graphed/584*100:.1f}%  (ch.15's engine reached 2.8%)")
    json.dump({"eager": eager, "graphed": graphed, "speedup": graphed/eager},
              open(f"{ROOT}/bench/results/command-recording.json", "w"), indent=1)


if __name__ == "__main__":
    main()
