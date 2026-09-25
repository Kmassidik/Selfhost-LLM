#!/usr/bin/env python3
"""R26 step 3 — does finer scaling granularity rescue post-training ternary?
Whole-matrix scalar vs per-row vs group-128 vs group-32 absmean scales. No training —
isolates the *scaling* lever (Bonsai uses group-128)."""
import torch, torch.nn as nn, math
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/root/Desktop/selfhostllm/models/hf/Qwen2.5-0.5B-Instruct"
dev = "cuda"
TEXT = ("Photosynthesis converts light into chemical energy inside chloroplasts that hold "
"chlorophyll; water is split into oxygen and the Calvin cycle fixes carbon dioxide into "
"sugars that feed nearly every food chain on Earth.") * 4

tok = AutoTokenizer.from_pretrained(MODEL)
def load(): return AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
def ppl(m):
    ids = tok(TEXT, return_tensors="pt").input_ids.to(dev)
    with torch.no_grad(): return math.exp(m(ids, labels=ids).loss.item())

def tern(x, s):  # ternary with given scale tensor s (broadcastable)
    return torch.clamp(torch.round(x / s), -1, 1) * s

@torch.no_grad()
def quant(W, mode):
    Wf = W.float(); out, inf = W.shape
    if mode == "scalar":
        s = Wf.abs().mean().clamp_min(1e-8);            return tern(Wf, s).to(W.dtype)
    if mode == "row":
        s = Wf.abs().mean(1, keepdim=True).clamp_min(1e-8); return tern(Wf, s).to(W.dtype)
    gs = mode
    if inf % gs:                                        # odd layer -> row scale
        s = Wf.abs().mean(1, keepdim=True).clamp_min(1e-8); return tern(Wf, s).to(W.dtype)
    G = Wf.reshape(out, inf // gs, gs)
    s = G.abs().mean(-1, keepdim=True).clamp_min(1e-8)
    return tern(G, s).reshape(out, inf).to(W.dtype)

@torch.no_grad()
def apply_(m, mode):
    for name, mod in m.named_modules():
        if isinstance(mod, nn.Linear) and "lm_head" not in name:
            mod.weight.data = quant(mod.weight.data, mode)

base = ppl(load())
print(f"fp16 baseline            : ppl {base:.2f}")
for label, mode in [("whole-matrix scalar", "scalar"), ("per-row", "row"),
                    ("group-128", 128), ("group-32", 32)]:
    m = load(); apply_(m, mode); p = ppl(m)
    print(f"ternary {label:20}: ppl {p:>16,.0f}   ({p/base:>12,.0f}x baseline)")
    del m; torch.cuda.empty_cache()
