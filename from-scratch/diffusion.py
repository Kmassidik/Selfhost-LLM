#!/usr/bin/env python3
"""From Scratch, Part B — a diffusion model by hand, on MNIST.
Learn to REVERSE a noising process: predict the noise added to an image, then sample by
denoising from pure noise. Renders generated digits as ASCII so you can see them."""
import os, math, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
dev = "cuda"; torch.manual_seed(0)

# ---- data: raw MNIST, parsed by hand, normalized to [-1,1] ----
with open("mnist/train-images-idx3-ubyte", "rb") as f:
    f.read(16); arr = np.frombuffer(f.read(), np.uint8).reshape(-1, 28, 28)
X = (torch.tensor(arr, dtype=torch.float32) / 127.5 - 1.0).unsqueeze(1)   # (N,1,28,28)
N = X.shape[0]

# ---- forward noising schedule (by hand) ----
T = 200
beta = torch.linspace(1e-4, 0.02, T, device=dev)
alpha = 1 - beta; abar = torch.cumprod(alpha, 0)

def time_emb(t, dim):                       # sinusoidal timestep embedding
    half = dim // 2
    fr = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[:, None].float() * fr[None]
    return torch.cat([a.sin(), a.cos()], -1)

# ---- denoiser: a small CNN with time conditioning (by hand) ----
class Net(nn.Module):
    def __init__(s, ch=96, td=64):
        super().__init__(); s.td = td
        s.temb = nn.Sequential(nn.Linear(td, ch), nn.SiLU(), nn.Linear(ch, ch))
        s.inc = nn.Conv2d(1, ch, 3, padding=1)
        s.c1 = nn.Conv2d(ch, ch, 3, padding=1); s.n1 = nn.GroupNorm(8, ch)
        s.c2 = nn.Conv2d(ch, ch, 3, padding=1); s.n2 = nn.GroupNorm(8, ch)
        s.c3 = nn.Conv2d(ch, ch, 3, padding=1); s.n3 = nn.GroupNorm(8, ch)
        s.out = nn.Conv2d(ch, 1, 3, padding=1)
    def forward(s, x, t):
        te = s.temb(time_emb(t, s.td))[:, :, None, None]
        h = F.silu(s.inc(x))
        h = F.silu(s.n1(s.c1(h)) + te)
        h = F.silu(s.n2(s.c2(h)) + te)
        h = F.silu(s.n3(s.c3(h)) + te)
        return s.out(h)

model = Net().to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
print(f"[cfg] {sum(p.numel() for p in model.parameters())/1e6:.2f}M params | {N} images | T={T}", flush=True)

# ---- train: predict the noise (MSE) ----
BATCH = 128; STEPS = int(os.environ.get("STEPS", "3000")); t0 = time.time()
for step in range(STEPS):
    x0 = X[torch.randint(0, N, (BATCH,))].to(dev)
    t = torch.randint(0, T, (BATCH,), device=dev)
    noise = torch.randn_like(x0)
    ab = abar[t][:, None, None, None]
    xt = ab.sqrt() * x0 + (1 - ab).sqrt() * noise         # forward diffusion
    loss = F.mse_loss(model(xt, t), noise)                 # predict the noise
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if step % 300 == 0: print(f"[step {step:4d}] mse {loss.item():.4f}", flush=True)
print(f"[done] {STEPS} steps in {time.time()-t0:.0f}s", flush=True)

# ---- sample: denoise from pure noise (DDPM reverse) ----
@torch.no_grad()
def sample(n):
    x = torch.randn(n, 1, 28, 28, device=dev)
    for i in reversed(range(T)):
        t = torch.full((n,), i, device=dev, dtype=torch.long)
        pred = model(x, t)
        a, ab, b = alpha[i], abar[i], beta[i]
        mean = (x - b / (1 - ab).sqrt() * pred) / a.sqrt()
        x = mean + (b.sqrt() * torch.randn_like(x) if i > 0 else 0)
    return x.clamp(-1, 1)

ramp = " .:-=+*#%@"
def show(img):                               # full 28x28 ASCII, one digit
    a = ((img[0] + 1) / 2)
    return ["".join(ramp[min(9, int(v * 10))] for v in row.tolist()) for row in a]

print("\n[samples] digits generated from pure noise (each is one 28x28 image):", flush=True)
imgs = sample(4)
for k in range(4):
    print(f"\n--- generated digit {k+1} ---", flush=True)
    for line in show(imgs[k]): print(line, flush=True)
