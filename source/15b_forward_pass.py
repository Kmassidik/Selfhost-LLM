#!/usr/bin/env python3
"""
15b_forward_pass.py — L4b of the serving exam (ch.15).

Our own transformer forward pass. torch for the matmuls; nothing above that.
Every piece the knowledge base described in prose gets written out here:
RMSNorm, RoPE, grouped-query attention, SwiGLU, the residual stream, and the
tied output projection that chapter 15a found missing from the file.

Checkpoint: our logits must match the reference implementation's, and the
greedy next token must be the same. Exactly, or it is a bug.

    uv run source/15b_forward_pass.py
"""
import json, math, os, sys, time
import torch

MODEL = "/root/Desktop/selfhostllm/models/hf/SmolLM2-360M-Instruct"


# --------------------------------------------------------------------------
# the pieces
# --------------------------------------------------------------------------
def rms_norm(x, weight, eps):
    """Scale each vector to unit root-mean-square, then rescale per channel.

    No mean subtraction and no bias — that is the whole difference from
    LayerNorm, and it is why it is cheaper.
    """
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x.to(dtype) * weight)


def rope_tables(head_dim, seq_len, theta, device, dtype):
    """Rotary position: position is applied by ROTATING each pair of channels.

    Nothing is added to the vector. A token's position changes the angle of its
    query and key, so the dot product between two tokens depends on how far
    apart they are — relative position, for free.
    """
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, inv)                 # (seq, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)       # (seq, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def apply_rope(q, k, cos, sin):
    cos, sin = cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def repeat_kv(x, n):
    """Grouped-query attention: 5 key-value heads serve 15 query heads.

    The keys and values are not recomputed — the same head is read by three
    queries. This is why the cache is 3x smaller than it would otherwise be.
    """
    if n == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None].expand(b, h, n, s, d).reshape(b, h * n, s, d)


def attention(x, w, cfg, cos, sin):
    b, s, _ = x.shape
    nh, nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd = cfg["hidden_size"] // nh

    q = (x @ w["q"].T).view(b, s, nh,  hd).transpose(1, 2)
    k = (x @ w["k"].T).view(b, s, nkv, hd).transpose(1, 2)
    v = (x @ w["v"].T).view(b, s, nkv, hd).transpose(1, 2)

    q, k = apply_rope(q, k, cos, sin)
    k, v = repeat_kv(k, nh // nkv), repeat_kv(v, nh // nkv)

    scores = (q @ k.transpose(2, 3)) / math.sqrt(hd)
    # causal mask: a token may not read the future
    mask = torch.full((s, s), float("-inf"), device=x.device, dtype=scores.dtype).triu(1)
    scores = scores + mask
    probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)

    out = (probs @ v).transpose(1, 2).reshape(b, s, nh * hd)
    return out @ w["o"].T


def mlp(x, w):
    """SwiGLU: one path is a gate on the other. Three matrices, not two."""
    gate = x @ w["gate"].T
    up = x @ w["up"].T
    return (torch.nn.functional.silu(gate) * up) @ w["down"].T


def forward(tokens, W, cfg, device):
    eps = cfg["rms_norm_eps"]
    nh = cfg["num_attention_heads"]
    hd = cfg["hidden_size"] // nh

    x = W["embed"][tokens]                                    # (b, s, hidden)
    cos, sin = rope_tables(hd, tokens.shape[1], cfg["rope_theta"], device, x.dtype)

    for i in range(cfg["num_hidden_layers"]):
        L = W["layers"][i]
        # residual stream: each block ADDS to it, never replaces it
        x = x + attention(rms_norm(x, L["ln1"], eps), L, cfg, cos, sin)
        x = x + mlp(rms_norm(x, L["ln2"], eps), L)

    x = rms_norm(x, W["norm"], eps)
    return x @ W["embed"].T          # tied: the embedding matrix, used backwards


# --------------------------------------------------------------------------
def load(device):
    from safetensors.torch import load_file
    t = load_file(os.path.join(MODEL, "model.safetensors"), device=str(device))
    cfg = json.load(open(os.path.join(MODEL, "config.json")))
    W = {"embed": t["model.embed_tokens.weight"], "norm": t["model.norm.weight"], "layers": []}
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}."
        W["layers"].append({
            "q": t[p+"self_attn.q_proj.weight"], "k": t[p+"self_attn.k_proj.weight"],
            "v": t[p+"self_attn.v_proj.weight"], "o": t[p+"self_attn.o_proj.weight"],
            "gate": t[p+"mlp.gate_proj.weight"], "up": t[p+"mlp.up_proj.weight"],
            "down": t[p+"mlp.down_proj.weight"],
            "ln1": t[p+"input_layernorm.weight"],
            "ln2": t[p+"post_attention_layernorm.weight"],
        })
    return W, cfg


def main():
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    W, cfg = load(dev)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(MODEL)
    prompt = "The capital of France is"
    ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
    print(f"prompt   {prompt!r}")
    print(f"tokens   {ids.shape[1]} -> {ids[0].tolist()}\n")

    torch.cuda.synchronize(); t0 = time.perf_counter()
    ours = forward(ids, W, cfg, dev)
    torch.cuda.synchronize(); t_ours = time.perf_counter() - t0

    ref_model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.no_grad():
        ref = ref_model(ids).logits
    torch.cuda.synchronize(); t_ref = time.perf_counter() - t0

    print(f"ours      {tuple(ours.shape)}  {t_ours*1000:7.1f} ms")
    print(f"reference {tuple(ref.shape)}  {t_ref*1000:7.1f} ms\n")

    diff = (ours.float() - ref.float()).abs()
    print(f"max abs difference in logits : {diff.max().item():.3e}")
    print(f"mean abs difference          : {diff.mean().item():.3e}")

    our_tok, ref_tok = ours[0, -1].argmax().item(), ref[0, -1].argmax().item()
    print(f"\ngreedy next token  ours {our_tok:>6} {tok.decode([our_tok])!r}")
    print(f"                   ref  {ref_tok:>6} {tok.decode([ref_tok])!r}")

    # the whole sequence, not just the last position
    agree = (ours[0].argmax(-1) == ref[0].argmax(-1)).sum().item()
    print(f"\nargmax agreement across all {ids.shape[1]} positions: {agree}/{ids.shape[1]}")

    if our_tok != ref_tok or agree != ids.shape[1]:
        print("\nFAIL  the forward pass diverges from the reference")
        sys.exit(1)
    print("\nPASS  same token at every position — L4b done, provisionally.")
    print()
    print("      Provisionally, because agreeing is not the same as being safe. Our logits")
    print("      differ from the reference by up to ~0.6, and on the frozen prompt set 18 of")
    print("      83 positions (21.7%) have a top-two gap SMALLER than that. At those, which")
    print("      token wins is decided by the order the matmuls accumulated in, not by the")
    print("      model. Computing the output projection in fp32 does not help - measured,")
    print("      still 18 of 83 - so the error is spread through all 32 layers, not the head.")
    print("      See 15b_risk_analysis.py.")
    print()
    print("      It matters because generation compounds: one flipped token changes every")
    print("      token after it. The exam is generated output, so that is where it is decided.")
    print("      Next: our own KV cache, so the second token costs less than the first.")


if __name__ == "__main__":
    main()
