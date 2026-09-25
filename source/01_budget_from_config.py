#!/usr/bin/env python3
"""Derive a model's parameter budget from its config.json alone — no weights.

    python3 budget_from_config.py <config.json> [<config.json> ...]
    python3 budget_from_config.py /root/Desktop/selfhostllm/models/configs/*/config.json

Validated twice: against Llama-3-8B's published config (all seven line items
exact) and against SmolLM2-360M's actual weight file, where it predicted the
file size to the byte. If the total here disagrees with a real file, the
architecture is not what the config implies — which is the point of running it.
"""
import json, sys, os


def budget(c):
    h = c["hidden_size"]; L = c["num_hidden_layers"]
    hds = c["num_attention_heads"]; kvh = c.get("num_key_value_heads", hds)
    ffn = c["intermediate_size"]; V = c["vocab_size"]
    hd = c.get("head_dim", h // hds)
    emb  = V * h
    attn = h*hds*hd + 2*(h*kvh*hd) + hds*hd*h
    mlp  = 3 * h * ffn
    per  = attn + mlp + 2*h
    head = 0 if c.get("tie_word_embeddings") else V * h
    return dict(emb=emb, attn=attn, mlp=mlp, per=per, layers=per*L, head=head,
                total=emb + per*L + h + head, h=h, L=L, hds=hds, kvh=kvh,
                ffn=ffn, V=V, hd=hd, tied=bool(c.get("tie_word_embeddings")))


def kv_per_token(b, dtype_bytes=2):
    return 2 * b["L"] * b["kvh"] * b["hd"] * dtype_bytes


def main():
    paths = sys.argv[1:]
    if not paths:
        sys.exit(__doc__)
    if len(paths) == 1:
        c = json.load(open(paths[0])); b = budget(c)
        name = os.path.basename(os.path.dirname(os.path.abspath(paths[0])))
        print(f"{name}\n")
        print(f"  hidden {b['h']} · {b['L']} layers · {b['hds']} heads / {b['kvh']} kv "
              f"· head_dim {b['hd']} · ffn {b['ffn']} · vocab {b['V']}"
              f"{' · tied embeddings' if b['tied'] else ''}\n")
        rows = [("embedding", b["emb"]), ("attention, one layer", b["attn"]),
                ("feed-forward, one layer", b["mlp"]), ("ONE LAYER", b["per"]),
                (f"x {b['L']} layers", b["layers"]), ("output head", b["head"]),
                ("TOTAL", b["total"])]
        for k, v in rows:
            print(f"  {k:<28}{v:>16,}")
        print()
        for lbl, bpp in (("bf16", 2), ("int8", 1), ("Q4_K_M", 0.55)):
            print(f"  weights @ {lbl:<8}{b['total']*bpp/1024**3:>8.2f} GiB")
        kv = kv_per_token(b)
        print(f"\n  KV cache: {kv:,} bytes/token ({kv/1024:.0f} KiB)")
        for ctx in (4096, 32768, 131072):
            print(f"    {ctx:>7,} tokens -> {kv*ctx/1024**3:>7.2f} GiB")
        return

    print(f"{'model':<24}{'params':>16}{'bf16':>9}{'Q4':>8}{'KV/tok':>10}  shape")
    for p in paths:
        try:
            c = json.load(open(p))
        except Exception:
            continue
        b = budget(c)
        name = os.path.basename(os.path.dirname(os.path.abspath(p)))
        print(f"{name:<24}{b['total']:>16,}{b['total']*2/1024**3:>8.1f}G"
              f"{b['total']*0.55/1024**3:>7.1f}G{kv_per_token(b)/1024:>9.0f}K"
              f"  h{b['h']} L{b['L']} {b['hds']}/{b['kvh']} ffn{b['ffn']} V{b['V']}"
              f"{' tied' if b['tied'] else ''}")


if __name__ == "__main__":
    main()
