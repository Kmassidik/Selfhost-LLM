#!/usr/bin/env python3
"""
28_the_run.py — training a model from nothing (ch.28).

Twenty-seven chapters of running somebody else's file of numbers. This makes
one, from random noise, on the hardware this guide has been measuring.

It is deliberately tiny. Chapter 26 priced the honest version at seventy-nine
days and chapter 27 found the corpus three hundred times too small, so the
ambition here is not a useful model — it is to watch the thing work end to end
and to be able to say exactly what happened.

The architecture is the one written by hand in chapter 15: RMSNorm, rotary
positions, grouped-query attention, SwiGLU, a tied output projection.

ON THE DATA: chapter 27 concluded this project's own writing must never be
trained on, because it is chapter 22's evaluation set. That ruling is about
models the AGENT harness will judge. This model is far too small to use a tool
and will never be scored by that harness; it is judged only by held-out loss on
text it has not seen. The split is real and enforced below.

    uv run source/28_the_run.py
"""
import glob, json, math, os, random, time
import torch
import torch.nn.functional as F

ROOT = "/root/Desktop/selfhostllm"
dev = torch.device("cuda:0")
torch.manual_seed(0); random.seed(0)

# ── the model, same shapes as chapter 15, scaled down ──────────────────
CFG = dict(vocab=49152, dim=256, layers=6, heads=8, kv_heads=2,
           hidden=768, seq=256, eps=1e-5, theta=10000.0)


class Block(torch.nn.Module):
    def __init__(s, c):
        super().__init__()
        d, h, kv = c["dim"], c["heads"], c["kv_heads"]
        s.hd = d // h; s.h = h; s.kv = kv
        s.q = torch.nn.Linear(d, h * s.hd, bias=False)
        s.k = torch.nn.Linear(d, kv * s.hd, bias=False)
        s.v = torch.nn.Linear(d, kv * s.hd, bias=False)
        s.o = torch.nn.Linear(h * s.hd, d, bias=False)
        s.gate = torch.nn.Linear(d, c["hidden"], bias=False)
        s.up = torch.nn.Linear(d, c["hidden"], bias=False)
        s.down = torch.nn.Linear(c["hidden"], d, bias=False)
        s.n1 = torch.nn.Parameter(torch.ones(d))
        s.n2 = torch.nn.Parameter(torch.ones(d))
        s.eps = c["eps"]

    def norm(s, x, w):
        return (x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + s.eps)).type_as(x) * w

    def forward(s, x, cos, sin):
        b, t, _ = x.shape
        h = s.norm(x, s.n1)
        q = s.q(h).view(b, t, s.h, s.hd).transpose(1, 2)
        k = s.k(h).view(b, t, s.kv, s.hd).transpose(1, 2)
        v = s.v(h).view(b, t, s.kv, s.hd).transpose(1, 2)
        def rot(z):
            z2 = torch.cat((-z[..., s.hd//2:], z[..., :s.hd//2]), -1)
            return z * cos + z2 * sin
        q, k = rot(q), rot(k)
        rep = s.h // s.kv
        k = k.repeat_interleave(rep, dim=1); v = v.repeat_interleave(rep, dim=1)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + s.o(a.transpose(1, 2).reshape(b, t, s.h * s.hd))
        h = s.norm(x, s.n2)
        return x + s.down(F.silu(s.gate(h)) * s.up(h))


class Tiny(torch.nn.Module):
    def __init__(s, c):
        super().__init__()
        s.c = c
        s.emb = torch.nn.Embedding(c["vocab"], c["dim"])
        s.blocks = torch.nn.ModuleList([Block(c) for _ in range(c["layers"])])
        s.nf = torch.nn.Parameter(torch.ones(c["dim"]))
        hd = c["dim"] // c["heads"]
        inv = 1.0 / (c["theta"] ** (torch.arange(0, hd, 2).float() / hd))
        pos = torch.arange(c["seq"]).float()
        f = torch.outer(pos, inv); e = torch.cat((f, f), -1)
        s.register_buffer("cos", e.cos()[None, None], persistent=False)
        s.register_buffer("sin", e.sin()[None, None], persistent=False)

    def forward(s, idx, targets=None):
        t = idx.shape[1]
        x = s.emb(idx)
        cos, sin = s.cos[:, :, :t], s.sin[:, :, :t]
        for b in s.blocks:
            x = b(x, cos, sin)
        x = (x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + s.c["eps"])).type_as(x) * s.nf
        logits = x @ s.emb.weight.T          # tied, as chapter 15a found
        if targets is None:
            return logits
        return logits, F.cross_entropy(logits.view(-1, s.c["vocab"]), targets.reshape(-1))


# ── the data, with a real held-out split ───────────────────────────────
def build_corpus(tok):
    files = []
    for d in ("source", "bench", "docs", "planning"):
        files += glob.glob(f"{ROOT}/{d}/**/*.*", recursive=True)
    files = sorted(f for f in files if os.path.splitext(f)[1] in {".py", ".md", ".json", ".sh"})
    random.Random(0).shuffle(files)
    cut = int(len(files) * 0.9)
    def ids(fs):
        out = []
        for f in fs:
            try: out += tok(open(f, errors="replace").read(), add_special_tokens=False).input_ids
            except Exception: pass
        return torch.tensor(out, dtype=torch.long)
    return ids(files[:cut]), ids(files[cut:]), len(files[:cut]), len(files[cut:])


def batch(data, bs, seq):
    i = torch.randint(0, len(data) - seq - 1, (bs,))
    x = torch.stack([data[j:j+seq] for j in i]).to(dev)
    y = torch.stack([data[j+1:j+seq+1] for j in i]).to(dev)
    return x, y


@torch.no_grad()
def held_out(m, data, seq, n=20):
    m.eval(); tot = 0.0
    for _ in range(n):
        x, y = batch(data, 8, seq)
        tot += m(x, y)[1].item()
    m.train()
    return tot / n


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(f"{ROOT}/models/hf/SmolLM2-360M-Instruct")
    tr, ho, ntr, nho = build_corpus(tok)

    m = Tiny(CFG).to(dev)
    P = sum(p.numel() for p in m.parameters())
    print("=" * 76)
    print("TRAINING A MODEL FROM NOTHING")
    print("=" * 76)
    print(f"  parameters        {P:,}")
    print(f"  architecture      {CFG['layers']} layers, dim {CFG['dim']}, "
          f"{CFG['heads']} heads / {CFG['kv_heads']} kv, seq {CFG['seq']}")
    print(f"  training tokens   {len(tr):,}  ({ntr} files)")
    print(f"  held-out tokens   {len(ho):,}  ({nho} files, never trained on)")
    print(f"  random-guess loss {math.log(CFG['vocab']):.3f}  (log of the vocabulary)")

    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.1)
    STEPS, BS = 3000, 16
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=STEPS, pct_start=0.1)
    print()
    print(f"  {'step':>6}{'train loss':>13}{'held-out':>12}{'tok seen':>13}{'elapsed':>10}")
    print("  " + "-" * 56)
    t0 = time.perf_counter(); hist = []
    for step in range(1, STEPS + 1):
        x, y = batch(tr, BS, CFG["seq"])
        loss = m(x, y)[1]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if step % 500 == 0 or step == 1:
            h = held_out(m, ho, CFG["seq"])
            seen = step * BS * CFG["seq"]
            print(f"  {step:>6}{loss.item():>13.3f}{h:>12.3f}{seen:>13,}"
                  f"{time.perf_counter()-t0:>9.0f}s")
            hist.append({"step": step, "train": loss.item(), "held_out": h, "seen": seen})

    print()
    print("=" * 76)
    print("WHAT IT LEARNED")
    print("=" * 76)
    first, last = hist[0], hist[-1]
    print(f"  held-out loss   {first['held_out']:.3f}  ->  {last['held_out']:.3f}")
    print(f"  perplexity      {math.exp(first['held_out']):>8,.0f}  ->  {math.exp(last['held_out']):>8,.1f}")
    print(f"  random guessing {math.exp(math.log(CFG['vocab'])):>8,.0f}")
    print()
    m.eval()
    for prompt in ("def read_", "The engine "):
        ids = torch.tensor([tok(prompt, add_special_tokens=False).input_ids], device=dev)
        for _ in range(40):
            nxt = m(ids[:, -CFG["seq"]:])[0, -1].argmax().view(1, 1)
            ids = torch.cat([ids, nxt], 1)
        print(f"  {prompt!r} -> {tok.decode(ids[0])[:110]!r}")
    json.dump({"params": P, "history": hist, "cfg": CFG},
              open(f"{ROOT}/bench/results/the-run.json", "w"), indent=1)


if __name__ == "__main__":
    main()
