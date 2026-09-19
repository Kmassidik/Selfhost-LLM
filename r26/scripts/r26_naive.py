#!/usr/bin/env python3
"""R26 step 1 — naive ternary rounding of Qwen2.5-0.5B, measured as the collapse.
BitNet-style absmean ternary: scale = mean(|W|); W_q = clamp(round(W/scale),-1,1)*scale."""
import torch, torch.nn as nn, math
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/root/Desktop/selfhostllm/models/hf/Qwen2.5-0.5B-Instruct"
dev = "cuda"

TEXT = (
"Photosynthesis is the process by which green plants, algae, and some bacteria convert "
"light energy into chemical energy stored in glucose. It takes place mainly in the leaves, "
"inside cell structures called chloroplasts, which contain the green pigment chlorophyll. "
"During the light-dependent reactions, chlorophyll absorbs sunlight and uses its energy to "
"split water molecules into oxygen, protons, and electrons. The oxygen is released into the "
"air as a by-product, while the energy is captured in two carrier molecules. In the second "
"stage, the Calvin cycle, that captured energy drives the conversion of carbon dioxide from "
"the atmosphere into sugars. These sugars fuel the plant's growth and are the foundation of "
"nearly every food chain on Earth. The overall reaction combines carbon dioxide and water, "
"powered by light, to produce glucose and oxygen. Because it removes carbon dioxide and "
"releases oxygen, photosynthesis also shapes the planet's climate and made complex animal "
"life possible in the first place."
) * 3

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(dev).eval()

def ppl():
    ids = tok(TEXT, return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        loss = model(ids, labels=ids).loss
    return math.exp(loss.item()), ids.shape[1]

base, ntok = ppl()
print(f"eval tokens         : {ntok}")
print(f"fp16 baseline ppl   : {base:.2f}")

@torch.no_grad()
def ternary_(m):
    nlin = zeros = total = 0
    for name, mod in m.named_modules():
        if isinstance(mod, nn.Linear) and "lm_head" not in name:
            W = mod.weight.data.float()
            scale = W.abs().mean()                       # BitNet b1.58 absmean scalar
            Wq = torch.clamp(torch.round(W / scale), -1, 1)
            zeros += (Wq == 0).sum().item(); total += Wq.numel()
            mod.weight.data = (Wq * scale).to(mod.weight.dtype)
            nlin += 1
    return nlin, zeros / total

nlin, sparsity = ternary_(model)
tern, _ = ppl()
print(f"quantized layers    : {nlin} Linear -> ternary (scalar absmean)")
print(f"ternary sparsity    : {sparsity*100:.1f}% of weights are exactly 0")
print(f"naive ternary ppl   : {tern:.2f}")
print(f"COLLAPSE            : {tern/base:.1f}x worse than fp16")
