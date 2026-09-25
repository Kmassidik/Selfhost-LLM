#!/usr/bin/env python3
"""From Scratch, Part C — audio is just another tokenization.
mu-law encode a waveform into 256 tokens, then train the SAME GPT from Part A to predict the
next audio sample. Generate audio autoregressively; check with an FFT whether it learned the
note frequencies. Ties the whole track together: sound -> tokens -> Part A's language model."""
import os, math, time, numpy as np, wave, torch, torch.nn as nn, torch.nn.functional as F
dev = "cuda"; torch.manual_seed(0)
SR, MU, V = 8000, 255, 256

# ---- synthesize a melody corpus (self-contained) ----
notes = [262, 330, 392, 523, 392, 330]        # C E G C G E, a little tune
dur = 0.15; sig = []
for _ in range(400):
    for fr in notes:
        n = int(SR * dur); t = np.arange(n) / SR
        env = np.clip(np.minimum(t / 0.01, (dur - t) / 0.02), 0, 1)
        sig.append(0.6 * np.sin(2 * np.pi * fr * t) * env)
wav = np.concatenate(sig).astype(np.float32)

def enc(x):
    y = np.sign(x) * np.log1p(MU * np.abs(x)) / np.log1p(MU)
    return np.clip(((y + 1) / 2 * MU + 0.5).astype(np.int64), 0, MU)
def dec(tk):
    y = tk.astype(np.float32) / MU * 2 - 1
    return np.sign(y) * (1 / MU) * ((1 + MU) ** np.abs(y) - 1)

tokens = torch.tensor(enc(wav), dtype=torch.long); Nt = len(tokens)
print(f"[cfg] corpus {Nt} samples ({Nt/SR:.1f}s) | vocab {V} | notes {notes}", flush=True)

# ---- the SAME GPT as Part A ----
BLOCK, DIM, HEADS, LAYERS = 512, 256, 4, 4
class Attn(nn.Module):
    def __init__(s): super().__init__(); s.qkv = nn.Linear(DIM, 3*DIM); s.proj = nn.Linear(DIM, DIM)
    def forward(s, x):
        B, T, C = x.shape; hd = C // HEADS
        q, k, v = s.qkv(x).split(C, 2)
        q = q.view(B, T, HEADS, hd).transpose(1, 2); k = k.view(B, T, HEADS, hd).transpose(1, 2); v = v.view(B, T, HEADS, hd).transpose(1, 2)
        a = (q @ k.transpose(-2, -1)) * (hd ** -0.5)
        a = a.masked_fill(torch.tril(torch.ones(T, T, device=x.device)) == 0, float("-inf"))
        return s.proj((F.softmax(a, -1) @ v).transpose(1, 2).reshape(B, T, C))
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
        logits = s.head(s.lnf(x)); loss = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1)) if tgt is not None else None
        return logits, loss
    @torch.no_grad()
    def gen(s, idx, m):
        for _ in range(m):
            l, _ = s(idx[:, -BLOCK:]); idx = torch.cat([idx, torch.multinomial(F.softmax(l[:, -1], -1), 1)], 1)
        return idx

model = GPT().to(dev); opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
print(f"[cfg] {sum(p.numel() for p in model.parameters())/1e6:.1f}M params (same GPT as Part A)", flush=True)
tok_dev = tokens.to(dev)
def batch(B=32):
    ix = torch.randint(0, Nt - BLOCK - 1, (B,))
    x = torch.stack([tok_dev[i:i+BLOCK] for i in ix]); y = torch.stack([tok_dev[i+1:i+1+BLOCK] for i in ix])
    return x, y

STEPS = int(os.environ.get("STEPS", "1500")); t0 = time.time()
for step in range(STEPS):
    x, y = batch(); _, loss = model(x, y)
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    if step % 250 == 0: print(f"[step {step:4d}] loss {loss.item():.3f}", flush=True)
print(f"[done] {STEPS} steps in {time.time()-t0:.0f}s", flush=True)

# ---- generate ~1.5 s of audio, seeded by real tokens ----
seed = tok_dev[:BLOCK].unsqueeze(0)
gen = model.gen(seed, 12000)[0, BLOCK:].cpu().numpy()
audio = dec(gen)

# ---- did it learn the notes? FFT of the generated audio ----
spec = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
freqs = np.fft.rfftfreq(len(audio), 1 / SR)
peaks = freqs[np.argsort(spec)[-6:]]
print(f"\n[FFT] dominant frequencies in GENERATED audio (Hz): {sorted(int(p) for p in peaks)}", flush=True)
print(f"[FFT] the melody's actual notes (Hz):                {sorted(set(notes))}", flush=True)

# ---- save WAV + a tiny ASCII waveform ----
pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
with wave.open("generated.wav", "w") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
print("[wav] wrote generated.wav", flush=True)
seg = audio[:400].reshape(20, 20).mean(1)                 # 20 points of the waveform
lo, hi = seg.min(), seg.max(); rows = 9
print("[waveform] first samples of generated audio:", flush=True)
for r in range(rows, -1, -1):
    line = "".join("#" if int((v - lo) / (hi - lo + 1e-9) * rows) == r else " " for v in seg)
    print("  " + line, flush=True)
