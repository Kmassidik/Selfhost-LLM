import json
c = json.load(open("pack-meta/config.json"))
t = c.get("text_config", c)
H  = t["hidden_size"]; L = t["num_hidden_layers"]; V = t["vocab_size"]
I  = t["intermediate_size"]; hd = t.get("head_dim"); 
nh = t.get("num_attention_heads"); nkv = t.get("num_key_value_heads")
lt = t.get("layer_types", [])
full = lt.count("full_attention"); lin = lt.count("linear_attention")
print(f"layers={L}  full_attn={full}  linear_attn={lin}  H={H}  I={I}  V={V}  heads={nh}/{nkv}  head_dim={hd}")
# rough param estimate (Qwen-style): embed + lm_head + per-layer(attn qkvo + mlp gate/up/down)
q = nh*hd; kv = nkv*hd
attn = H*q + 2*(H*kv) + q*H          # q,k,v,o  (approx; linear-attn differs but same order)
mlp  = 3*(H*I)                        # gate, up, down
per_layer = attn + mlp
embed = V*H
lm_head = V*H                         # assume untied
total = embed + lm_head + L*per_layer
print(f"approx params: {total/1e9:.2f} B")
pack_bytes = 8.61e9
print(f"pack on disk : 8.61 GB")
print(f"effective bits/weight (whole pack / params): {pack_bytes*8/total:.2f} bit")
print(f"claimed: 1.72 bit  |  tweet size 5.9 GB  |  actual 8.61 GB")
