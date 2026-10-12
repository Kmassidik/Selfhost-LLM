#!/usr/bin/env python3
"""Derive a model's parameter budget from its config.json alone — no weights.

    python3 budget_from_config.py <config.json> [<config.json> ...]
    python3 budget_from_config.py /root/Desktop/selfhostllm/models/configs/*/config.json

Validated against real files: Llama-3-8B's published config (all seven line
items exact), SmolLM2-360M's weight file (to the byte), Qwen3-8B and Qwen2.5-7B
against their published safetensors sizes, and Qwen3.5-0.8B, Qwen3.5-27B (via
Ternary Bonsai 2 27B, same shape) and Qwen3-Coder-30B-A3B against their GGUF
tensor counts — every one exact to the parameter. If the total here disagrees
with a real file, the architecture is not what the config implies — which is
the point of running it.

Three 2025-26 architectures are handled beyond the plain dense transformer:
  * multimodal wrappers (Qwen3.5): the language model sits in `text_config`;
    only it is counted, not the vision tower
  * hybrid attention (Qwen3.5): most layers are linear attention (Gated DeltaNet)
    with a fixed-size recurrent state instead of a per-token KV cache
  * mixture of experts (Qwen3-MoE): every expert is stored, only top-k run —
    so a model has a stored size and a smaller *active* size
"""
import json, sys, os


def budget(c):
    t = c.get("text_config", c)
    tied = bool(c.get("tie_word_embeddings", t.get("tie_word_embeddings", False)))
    h = t["hidden_size"]; L = t["num_hidden_layers"]
    hds = t["num_attention_heads"]; kvh = t.get("num_key_value_heads", hds)
    ffn = t.get("intermediate_size", 0); V = t["vocab_size"]
    hd = t.get("head_dim") or h // hds
    mt = t.get("model_type", "")
    qk_norm = mt.startswith("qwen3")                        # RMSNorm on q and k, per head
    gate = bool(t.get("attn_output_gate"))                  # q_proj also emits an output gate
    if mt.startswith("qwen2"):                              # Qwen2/2.5: bias on q, k, v only
        bias = hds*hd + 2*kvh*hd
    elif t.get("attention_bias"):                           # e.g. some Llama variants: q, k, v, o
        bias = hds*hd + 2*kvh*hd + h
    else:
        bias = 0

    # attention layer (full softmax attention, keeps a KV cache)
    attn = (h*hds*hd*(2 if gate else 1) + 2*(h*kvh*hd) + hds*hd*h
            + (2*hd if qk_norm else 0) + bias)

    # linear-attention layer (Gated DeltaNet): fixed-size state, no KV cache
    types = t.get("layer_types") or ["full_attention"] * L
    n_lin = sum(1 for x in types if x == "linear_attention"); n_att = L - n_lin
    lin = state = 0
    if n_lin:
        dk, dv = t["linear_key_head_dim"], t["linear_value_head_dim"]
        nk, nv = t["linear_num_key_heads"], t["linear_num_value_heads"]
        kd, vd, kern = nk*dk, nv*dv, t["linear_conv_kernel_dim"]
        conv = 2*kd + vd
        lin = (h*conv + h*vd + 2*h*nv      # in-projections: q,k,v | output gate z | beta, a
               + conv*kern                 # short causal depthwise conv
               + 2*nv + dv                 # dt_bias, A_log, gated RMSNorm
               + vd*h)                     # out projection
        state = n_lin * (nv*dk*dv + conv*(kern - 1)) * 4   # fp32 recurrent + conv state, bytes

    # feed-forward: dense, or mixture of experts
    E = t.get("num_experts", 0)
    if E:
        mf, k = t["moe_intermediate_size"], t["num_experts_per_tok"]
        sh = t.get("shared_expert_intermediate_size", 0)
        extra = 3*h*sh + (h if sh else 0)                   # shared expert + its gate
        mlp = E*3*h*mf + h*E + extra
        mlp_act = k*3*h*mf + h*E + extra
    else:
        mlp = mlp_act = 3*h*ffn

    emb = V*h
    head = 0 if tied else V*h
    norms = 2*h
    layers = n_att*attn + n_lin*lin + L*(mlp + norms)
    total = emb + layers + h + head
    active = total - L*(mlp - mlp_act)
    return dict(emb=emb, attn=attn, lin=lin, mlp=mlp, mlp_act=mlp_act, layers=layers,
                head=head, total=total, active=active, state=state, h=h, L=L,
                n_att=n_att, n_lin=n_lin, hds=hds, kvh=kvh, ffn=ffn, V=V, hd=hd,
                E=E, k=t.get("num_experts_per_tok", 0), mf=t.get("moe_intermediate_size", 0),
                tied=tied)


def kv_per_token(b, dtype_bytes=2):
    return 2 * b["n_att"] * b["kvh"] * b["hd"] * dtype_bytes   # only attention layers cache K,V


def shape(b):
    s = f"h{b['h']} L{b['L']}"
    if b["n_lin"]:
        s += f"({b['n_att']}attn+{b['n_lin']}lin)"
    s += f" {b['hds']}/{b['kvh']}"
    s += f" moe{b['E']}x{b['mf']} top{b['k']}" if b["E"] else f" ffn{b['ffn']}"
    return s


def main():
    paths = sys.argv[1:]
    if not paths:
        sys.exit(__doc__)
    if len(paths) == 1:
        c = json.load(open(paths[0])); b = budget(c)
        name = os.path.basename(os.path.dirname(os.path.abspath(paths[0])))
        print(f"{name}\n")
        print(f"  hidden {b['h']} · {b['L']} layers · {b['hds']} heads / {b['kvh']} kv "
              f"· head_dim {b['hd']} · vocab {b['V']}"
              f"{' · tied embeddings' if b['tied'] else ''}")
        if b["n_lin"]:
            print(f"  hybrid: {b['n_att']} attention + {b['n_lin']} linear-attention layers")
        if b["E"]:
            print(f"  mixture of experts: {b['E']} experts of {b['mf']}, top-{b['k']} per token")
        print()
        rows = [("embedding", b["emb"]), ("attention, one layer", b["attn"])]
        if b["n_lin"]:
            rows.append(("linear attention, one layer", b["lin"]))
        rows += [("feed-forward, one layer", b["mlp"])]
        if b["E"]:
            rows.append(("  of which active per token", b["mlp_act"]))
        rows += [(f"all {b['L']} layers", b["layers"]), ("output head", b["head"]),
                 ("TOTAL", b["total"])]
        if b["E"]:
            rows.append(("ACTIVE per token", b["active"]))
        for k, v in rows:
            print(f"  {k:<30}{v:>16,}")
        print()
        for lbl, bpp in (("bf16", 2), ("int8", 1), ("Q4_K_M", 0.55)):
            print(f"  weights @ {lbl:<8}{b['total']*bpp/1024**3:>8.2f} GiB")
        kv = kv_per_token(b)
        print(f"\n  KV cache: {kv:,} bytes/token ({kv/1024:.0f} KiB)"
              + (f" — {b['n_att']} of {b['L']} layers" if b["n_lin"] else ""))
        for ctx in (4096, 32768, 131072, 262144):
            print(f"    {ctx:>7,} tokens -> {kv*ctx/1024**3:>7.2f} GiB")
        if b["state"]:
            print(f"  + recurrent state: {b['state']/1024**2:.0f} MiB per sequence, "
                  f"whatever its length")
        return

    print(f"{'model':<30}{'params':>16}{'active':>9}{'bf16':>8}{'Q4':>7}{'KV/tok':>8}"
          f"{'state':>8}  shape")
    for p in paths:
        try:
            c = json.load(open(p))
        except Exception:
            continue
        b = budget(c)
        name = os.path.basename(os.path.dirname(os.path.abspath(p)))
        act = f"{b['active']/1e9:.1f}B" if b["E"] else "all"
        st = f"{b['state']/1024**2:.0f}M" if b["state"] else "-"
        print(f"{name:<30}{b['total']:>16,}{act:>9}{b['total']*2/1024**3:>7.1f}G"
              f"{b['total']*0.55/1024**3:>6.1f}G{kv_per_token(b)/1024:>7.0f}K{st:>8}"
              f"  {shape(b)}")


if __name__ == "__main__":
    main()
