#!/usr/bin/env python3
"""From Scratch, Part A — a GPT built by hand, trained on TinyStories (char-level).
Runs single-GPU or multi-GPU (DDP via torchrun). Reports throughput for the 1-vs-3 comparison.
Attention is written out, not imported (the house rule)."""
import os, time, math, torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

BLOCK, BATCH, DIM, HEADS, LAYERS = 256, 32, 384, 6, 6
STEPS = int(os.environ.get("STEPS", "1200")); LR = 3e-4
DATA = "corpus.txt"

# ---- DDP ----
ddp = "RANK" in os.environ
if ddp:
    dist.init_process_group("nccl")
    rank = int(os.environ["RANK"]); loc = int(os.environ["LOCAL_RANK"]); world = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{loc}"; torch.cuda.set_device(loc); master = rank == 0
else:
    rank = 0; world = 1; device = "cuda"; master = True
torch.manual_seed(1337 + rank)

# ---- data: char-level tokenizer, by hand ----
text = open(DATA).read()
chars = sorted(set(text)); V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}; itos = {i: c for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.95 * len(data)); tr, va = data[:n], data[n:]
def batch(split):
    d = tr if split == "train" else va
    ix = torch.randint(len(d) - BLOCK, (BATCH,))
    x = torch.stack([d[i:i+BLOCK] for i in ix]).to(device)
    y = torch.stack([d[i+1:i+1+BLOCK] for i in ix]).to(device)
    return x, y

# ---- model: transformer, by hand ----
class Attn(nn.Module):
    def __init__(s): super().__init__(); s.qkv = nn.Linear(DIM, 3*DIM); s.proj = nn.Linear(DIM, DIM)
    def forward(s, x):
        B, T, C = x.shape; hd = C // HEADS
        q, k, v = s.qkv(x).split(C, 2)
        q = q.view(B, T, HEADS, hd).transpose(1, 2); k = k.view(B, T, HEADS, hd).transpose(1, 2); v = v.view(B, T, HEADS, hd).transpose(1, 2)
        a = (q @ k.transpose(-2, -1)) * (hd ** -0.5)
        a = a.masked_fill(torch.tril(torch.ones(T, T, device=x.device)) == 0, float("-inf"))
        y = (F.softmax(a, -1) @ v).transpose(1, 2).reshape(B, T, C)
        return s.proj(y)
class Block(nn.Module):
    def __init__(s):
        super().__init__(); s.l1 = nn.LayerNorm(DIM); s.at = Attn(); s.l2 = nn.LayerNorm(DIM)
        s.mlp = nn.Sequential(nn.Linear(DIM, 4*DIM), nn.GELU(), nn.Linear(4*DIM, DIM))
    def forward(s, x): x = x + s.at(s.l1(x)); return x + s.mlp(s.l2(x))
class GPT(nn.Module):
    def __init__(s):
        super().__init__(); s.tok = nn.Embedding(V, DIM); s.pos = nn.Embedding(BLOCK, DIM)
        s.blocks = nn.ModuleList([Block() for _ in range(LAYERS)]); s.lnf = nn.LayerNorm(DIM); s.head = nn.Linear(DIM, V, bias=False)
    def forward(s, idx, tgt=None):
        B, T = idx.shape; x = s.tok(idx) + s.pos(torch.arange(T, device=idx.device))
        for b in s.blocks: x = b(x)
        logits = s.head(s.lnf(x)); loss = None
        if tgt is not None: loss = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1))
        return logits, loss
    @torch.no_grad()
    def gen(s, idx, m):
        for _ in range(m):
            l, _ = s(idx[:, -BLOCK:]); idx = torch.cat([idx, torch.multinomial(F.softmax(l[:, -1], -1), 1)], 1)
        return idx

model = GPT().to(device)
nparam = sum(p.numel() for p in model.parameters())
if master: print(f"[cfg] {nparam/1e6:.1f}M params | vocab {V} | world {world} | batch/gpu {BATCH} | block {BLOCK}", flush=True)
if ddp: model = DDP(model, device_ids=[loc])
raw = model.module if ddp else model
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1, betas=(0.9, 0.95))

@torch.no_grad()
def val():
    raw.eval(); L = sum(raw(*batch("val"))[1].item() for _ in range(20)) / 20; raw.train(); return L

# ---- train ----
torch.cuda.synchronize(); warm = 30; toks = 0; tstart = time.time()
for step in range(STEPS):
    if step == warm: torch.cuda.synchronize(); tstart = time.time(); toks = 0
    x, y = batch("train"); _, loss = model(x, y)
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    toks += BATCH * BLOCK * world
    if step % 200 == 0 and master: print(f"[step {step:4d}] train_loss {loss.item():.3f}", flush=True)
torch.cuda.synchronize()
if master:
    dt = time.time() - tstart; tps = toks / dt
    print(f"[RESULT] {world}-GPU  throughput {tps:,.0f} tok/s  |  val_loss {val():.3f}  |  {dt:.1f}s / {STEPS-warm} steps", flush=True)
    ctx = torch.tensor([[stoi["O"]]], device=device)
    s = "".join(itos[i] for i in raw.gen(ctx, 400)[0].tolist())
    print("[sample] " + s.replace("\n", " / "), flush=True)
if ddp: dist.destroy_process_group()
