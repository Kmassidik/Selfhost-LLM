"""Is the divergence bad math, or a different kernel doing the same math?

The reference calls torch's fused scaled_dot_product_attention. Ours writes the
same formula out by hand. Both are correct; they accumulate in different orders.
If the kernel is the cause, using the same one should close the gap — and
scaled_dot_product_attention is torch, which the rules allow.
"""
import json, math, importlib.util, torch, sys
HERE = "/root/Desktop/selfhostllm/source"
spec = importlib.util.spec_from_file_location("fwd", HERE + "/15b_forward_pass.py")
fwd = importlib.util.module_from_spec(spec); spec.loader.exec_module(fwd)

dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
W, cfg = fwd.load(dev)
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(fwd.MODEL)
ref_model = AutoModelForCausalLM.from_pretrained(fwd.MODEL, dtype=torch.bfloat16).to(dev).eval()


def attention_sdpa(x, w, cfg, cos, sin):
    """Identical mathematics, torch's fused kernel instead of our own steps."""
    b, s, _ = x.shape
    nh, nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd = cfg["hidden_size"] // nh
    q = (x @ w["q"].T).view(b, s, nh,  hd).transpose(1, 2)
    k = (x @ w["k"].T).view(b, s, nkv, hd).transpose(1, 2)
    v = (x @ w["v"].T).view(b, s, nkv, hd).transpose(1, 2)
    q, k = fwd.apply_rope(q, k, cos, sin)
    k, v = fwd.repeat_kv(k, nh // nkv), fwd.repeat_kv(v, nh // nkv)
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=(s > 1))
    return out.transpose(1, 2).reshape(b, s, nh * hd) @ w["o"].T


def forward_sdpa(tokens, W, cfg, device):
    eps, nh = cfg["rms_norm_eps"], cfg["num_attention_heads"]
    hd = cfg["hidden_size"] // nh
    x = W["embed"][tokens]
    cos, sin = fwd.rope_tables(hd, tokens.shape[1], cfg["rope_theta"], device, x.dtype)
    for i in range(cfg["num_hidden_layers"]):
        L = W["layers"][i]
        x = x + attention_sdpa(fwd.rms_norm(x, L["ln1"], eps), L, cfg, cos, sin)
        x = x + fwd.mlp(fwd.rms_norm(x, L["ln2"], eps), L)
    return fwd.rms_norm(x, W["norm"], eps) @ W["embed"].T


def expand(t):
    if not t.startswith("REPEAT:"): return t
    _, n, rest = t.split(":", 2); f, tail = rest.split("\n\n", 1)
    return f * int(n) + "\n\n" + tail

cases = [(p["id"], tok.apply_chat_template([{"role":"user","content":expand(p["text"])}],
          tokenize=False, add_generation_prompt=True))
         for p in json.load(open("/root/Desktop/selfhostllm/bench/prompts.json"))["prompts"]]

print(f"{'prompt':<10}{'pos':>6}{'hand-written':>26}{'fused (SDPA)':>26}")
print(f"{'':<10}{'':>6}{'max dlogit':>13}{'at risk':>13}{'max dlogit':>13}{'at risk':>13}")
print("-" * 68)
ta = tb = tn = 0
for name, text in cases:
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
    if ids.shape[1] > 1400: ids = ids[:, :1400]
    with torch.no_grad():
        a = fwd.forward(ids, W, cfg, dev)
        b = forward_sdpa(ids, W, cfg, dev)
        r = ref_model(ids).logits
    n = ids.shape[1]
    top2 = r[0].float().topk(2, dim=-1).values
    margin = top2[:, 0] - top2[:, 1]
    row = [name, n]
    for cand in (a, b):
        err = (cand[0].float() - r[0].float()).abs().max(dim=-1).values
        row += [err.max().item(), (margin < err).sum().item()]
    ta += row[3]; tb += row[5]; tn += n
    print(f"{row[0]:<10}{row[1]:>6}{row[2]:>13.3e}{row[3]:>8}/{n:<4}{row[4]:>13.3e}{row[5]:>8}/{n:<4}")
print("-" * 68)
print(f"{'TOTAL':<10}{tn:>6}{'':>13}{ta:>8}/{tn:<4}{'':>13}{tb:>8}/{tn:<4}")
print(f"\nat-risk: hand-written {ta/tn*100:.1f}%  ->  fused {tb/tn*100:.1f}%")
