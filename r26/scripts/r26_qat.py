#!/usr/bin/env python3
"""R26 step 2 — quantization-aware training (QAT) recovers the collapsed ternary model.
BitLinear: forward uses ternary fake-quant; a straight-through estimator (STE) lets
gradients flow to fp32 master weights, which learn to survive the rounding. Train on
general text, evaluate on HELD-OUT text (not trained on)."""
import torch, torch.nn as nn, torch.nn.functional as F, math
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/root/Desktop/selfhostllm/models/hf/Qwen2.5-0.5B-Instruct"
dev = "cuda"

# held-out eval text (NOT in training corpus)
EVAL = ("Photosynthesis is the process by which green plants and algae convert light energy "
"into chemical energy stored in glucose, inside chloroplasts that contain chlorophyll. "
"Light-dependent reactions split water into oxygen, protons and electrons; the Calvin cycle "
"then fixes carbon dioxide into sugars that feed nearly every food chain on Earth.") * 3

# training corpus — general knowledge, different topics from the eval
TRAIN = ("The Roman Empire was one of the largest empires in ancient history, spanning three "
"continents at its height. Roads, aqueducts and a common legal code bound its provinces "
"together. Volcanoes form where molten rock rises through the Earth's crust; when pressure "
"releases, lava, ash and gas erupt at the surface. The printing press, developed in the "
"fifteenth century, spread books and literacy across Europe and reshaped how knowledge "
"moved between people. Ocean currents redistribute heat around the planet, carrying warm "
"water toward the poles and cooling the tropics, which steadies regional climates. A market "
"economy coordinates the choices of many buyers and sellers through prices, which rise when "
"goods are scarce and fall when they are plentiful. Bees pollinate flowering plants as they "
"gather nectar, a quiet exchange that underpins much of the world's agriculture. Bridges "
"carry load by turning the downward pull of gravity into forces of tension and compression "
"that the structure's materials can safely bear.") * 6

tok = AutoTokenizer.from_pretrained(MODEL)

class STETernary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w):
        scale = w.abs().mean()
        return torch.clamp(torch.round(w / scale), -1, 1) * scale
    @staticmethod
    def backward(ctx, g):
        return g                       # straight-through: pretend quant was identity

class BitLinear(nn.Linear):
    def forward(self, x):
        return F.linear(x, STETernary.apply(self.weight), self.bias)

def to_bitlinear(m):
    n = 0
    for name, mod in list(m.named_modules()):
        for child_name, child in list(mod.named_children()):
            if isinstance(child, nn.Linear) and "lm_head" not in f"{name}.{child_name}":
                bl = BitLinear(child.in_features, child.out_features, bias=child.bias is not None)
                bl.weight = child.weight; bl.bias = child.bias
                setattr(mod, child_name, bl); n += 1
    return n

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev)
nlin = to_bitlinear(model)
model.gradient_checkpointing_enable()

def ppl():
    model.eval()
    ids = tok(EVAL, return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        return math.exp(model(ids, labels=ids).loss.item())

print(f"BitLinear layers    : {nlin}")
print(f"step   0 (naive)    : ppl {ppl():.1f}")

ids = tok(TRAIN, return_tensors="pt").input_ids.to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)  # bf16 states fit 8GB; normalizes big init grads
model.train()
STEPS, SEQ = 300, 256
for step in range(1, STEPS + 1):
    i = torch.randint(0, max(1, ids.shape[1] - SEQ), (1,)).item()
    chunk = ids[:, i:i + SEQ]
    loss = model(chunk, labels=chunk).loss
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # stop the explosion
    opt.step()
    if step in (25, 50, 100, 200, 300):
        print(f"step {step:>3} (QAT)      : ppl {ppl():.1f}   train_loss {loss.item():.3f}")
        model.train()
print("done")
