"""How many positions are one rounding error away from picking a different token?

A position is AT RISK when the gap between the reference's top two candidates is
smaller than the error between our logits and the reference's. There, which token
wins is decided by accumulation order, not by the model.
"""
import json, sys, torch, importlib.util
spec = importlib.util.spec_from_file_location("fwd", "/root/Desktop/selfhostllm/source/15b_forward_pass.py")
fwd = importlib.util.module_from_spec(spec); spec.loader.exec_module(fwd)

dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
W, cfg = fwd.load(dev)
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(fwd.MODEL)
ref_model = AutoModelForCausalLM.from_pretrained(fwd.MODEL, dtype=torch.bfloat16).to(dev).eval()
cases = [(p["id"], p["text"]) for p in json.load(open("/root/Desktop/selfhostllm/bench/prompts.json"))["prompts"]]


def fwd_fp32_head(tokens, W, cfg, device):
    """Same forward pass, but the final projection accumulates in fp32."""
    eps, nh = cfg["rms_norm_eps"], cfg["num_attention_heads"]
    hd = cfg["hidden_size"] // nh
    x = W["embed"][tokens]
    cos, sin = fwd.rope_tables(hd, tokens.shape[1], cfg["rope_theta"], device, x.dtype)
    for i in range(cfg["num_hidden_layers"]):
        L = W["layers"][i]
        x = x + fwd.attention(fwd.rms_norm(x, L["ln1"], eps), L, cfg, cos, sin)
        x = x + fwd.mlp(fwd.rms_norm(x, L["ln2"], eps), L)
    x = fwd.rms_norm(x, W["norm"], eps)
    return x.float() @ W["embed"].float().T


print(f"{'prompt':<10}{'pos':>5}{'bf16 head':>26}{'fp32 head':>26}")
print(f"{'':<10}{'':>5}{'dlogit':>12}{'at risk':>14}{'dlogit':>12}{'at risk':>14}")
print("-" * 67)
tot_a = tot_b = tot_n = 0
for name, text in cases:
    ids = tok(text, return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        a = fwd.forward(ids, W, cfg, dev)
        b = fwd_fp32_head(ids, W, cfg, dev)
        r = ref_model(ids).logits
    n = ids.shape[1]
    top2 = r[0].float().topk(2, dim=-1).values
    margin = top2[:, 0] - top2[:, 1]                       # per position
    out = [name, n]
    for cand in (a, b):
        err = (cand[0].float() - r[0].float()).abs().max(dim=-1).values   # per position
        risk = (margin < err).sum().item()
        out += [err.max().item(), risk]
    tot_a += out[3]; tot_b += out[5]; tot_n += n
    print(f"{out[0]:<10}{out[1]:>5}{out[2]:>12.3e}{out[3]:>7}/{n:<6}{out[4]:>12.3e}{out[5]:>7}/{n:<6}")
print("-" * 67)
print(f"{'TOTAL':<10}{tot_n:>5}{'':>12}{tot_a:>7}/{tot_n:<6}{'':>12}{tot_b:>7}/{tot_n:<6}")
print(f"\nat-risk positions: bf16 head {tot_a/tot_n*100:.1f}%  ->  fp32 head {tot_b/tot_n*100:.1f}%")
