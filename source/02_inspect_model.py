#!/usr/bin/env python3
"""Open a safetensors model file and count what is inside.

No model loading, no GPU, no torch. A safetensors file begins with an 8-byte
length followed by a JSON header describing every tensor — so the file can be
read as a catalogue without allocating a single weight. A 400 GB model can be
inspected this way in milliseconds.

    python3 inspect_model.py <path-to-model.safetensors> [--predict config.json]

With --predict, the expected parameter count is derived from the config first,
then compared against what the file actually contains. A mismatch means the
mental model of the architecture is wrong, which is the point of running it.
"""
import json, struct, os, sys, collections


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return n, json.loads(f.read(n))


def predict(cfg):
    h   = cfg["hidden_size"]
    L   = cfg["num_hidden_layers"]
    hds = cfg["num_attention_heads"]
    kvh = cfg.get("num_key_value_heads", hds)
    ffn = cfg["intermediate_size"]
    V   = cfg["vocab_size"]
    hd  = h // hds
    emb  = V * h
    attn = h*hds*hd + 2*(h*kvh*hd) + hds*hd*h
    mlp  = 3 * h * ffn
    per  = attn + mlp + 2*h
    head = 0 if cfg.get("tie_word_embeddings") else V * h
    return emb + per*L + h + head


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    path = sys.argv[1]
    n, header = read_header(path)
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}

    total, kinds, counts, shapes = 0, collections.Counter(), collections.Counter(), {}
    for name, meta in tensors.items():
        c = 1
        for d in meta["shape"]:
            c *= d
        total += c
        kind = ".".join(p for p in name.replace("model.layers.", "L.").split(".")
                        if not p.isdigit())
        kinds[kind] += c
        counts[kind] += 1
        shapes.setdefault(kind, meta["shape"])

    print(f"{os.path.basename(path)}\n")
    print(f"{'tensor':<36}{'n':>4}{'shape':>22}{'params':>16}")
    print("-" * 78)
    for k in sorted(kinds, key=lambda x: -kinds[x]):
        print(f"{k:<36}{counts[k]:>4}{str(shapes[k]):>22}{kinds[k]:>16,}")
    print("-" * 78)
    print(f"{'TOTAL':<36}{len(tensors):>4}{'':>22}{total:>16,}")

    dt = collections.Counter(m["dtype"] for m in tensors.values())
    size = os.path.getsize(path)
    bpp = size / total
    print(f"\ndtypes       : {dict(dt)}")
    print(f"file size    : {size:,} bytes")
    print(f"header       : {n + 8:,} bytes of JSON")
    print(f"bytes/param  : {bpp:.4f}")

    if "--predict" in sys.argv:
        cfg_path = sys.argv[sys.argv.index("--predict") + 1]
        p = predict(json.load(open(cfg_path)))
        print(f"\npredicted    : {p:,}")
        print(f"actual       : {total:,}")
        d = total - p
        print(f"difference   : {d:,}" + ("   EXACT" if d == 0 else "   <- the model is not what you think"))


if __name__ == "__main__":
    main()
