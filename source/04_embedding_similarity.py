"""Is meaning actually present in the embedding numbers? Measure it.

Loads only the embedding tensor from a model file — one tensor of hundreds — and
asks whether words that mean similar things sit near each other. Nothing is
trained and nothing is run; this is arithmetic on stored numbers.

    python3 probe_embedding.py <model-dir>
    SELFHOSTLLM_ROOT=/root/Desktop/selfhostllm python3 probe_embedding.py

Needs model.safetensors and tokenizer.json in the directory.
"""
import json, os, struct, sys, numpy as np
from tokenizers import Tokenizer

if len(sys.argv) > 1:
    D = sys.argv[1]
else:
    root = os.environ.get("SELFHOSTLLM_ROOT")
    if not root:
        sys.exit("give a model directory, or set SELFHOSTLLM_ROOT\n" + __doc__)
    D = os.path.join(root, "models/hf/SmolLM2-360M-Instruct")
if not os.path.exists(f"{D}/model.safetensors"):
    sys.exit(f"no model.safetensors in {D}")
tok = Tokenizer.from_file(f"{D}/tokenizer.json")

# pull just the embedding tensor out of the file, by its offset
with open(f"{D}/model.safetensors", "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
    m = hdr["model.embed_tokens.weight"]
    s, e = m["data_offsets"]
    f.seek(8 + n + s)
    raw = f.read(e - s)

u16 = np.frombuffer(raw, dtype="<u2")
E = (u16.astype(np.uint32) << 16).view(np.float32).reshape(m["shape"])
print(f"embedding tensor: {E.shape}, {E.nbytes/1e6:.0f} MB of the 724 MB file\n")

norm = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

def tid(w):
    ids = tok.encode(w, add_special_tokens=False).ids
    return ids[0] if len(ids) == 1 else None

def sim(a, b):
    ia, ib = tid(a), tid(b)
    if ia is None or ib is None: return None
    return float(norm[ia] @ norm[ib])

print("=== cosine similarity between single tokens ===")
pairs = [(" king"," queen"), (" king"," man"), (" man"," woman"),
         (" Paris"," London"), (" Paris"," France"), (" Paris"," banana"),
         (" cat"," dog"), (" cat"," kitten"), (" cat"," algebra"),
         (" three"," four"), (" three"," elephant"),
         (" happy"," joyful"), (" happy"," sad"), (" happy"," concrete")]
for a,b in pairs:
    v = sim(a,b)
    if v is not None:
        bar = "#" * max(0,int(v*40))
        print(f"  {a.strip():<9} {b.strip():<10} {v:>6.3f}  {bar}")

print("\n=== nearest neighbours in 960-dimensional space ===")
for w in [" king", " Monday", " seven", " Paris", " red"]:
    i = tid(w)
    if i is None: continue
    sims = norm @ norm[i]
    sims[i] = -1
    top = np.argsort(-sims)[:8]
    out = ", ".join(f"{tok.decode([int(j)]).strip()!r}" for j in top)
    print(f"  {w.strip():<8} -> {out}")

print("\n=== the classic analogy test ===")
for a,b,c in [(" king"," man"," woman"), (" Paris"," France"," Italy"),
              (" walking"," walk"," swim")]:
    ids=[tid(x) for x in (a,b,c)]
    if any(i is None for i in ids): continue
    v = E[ids[0]] - E[ids[1]] + E[ids[2]]
    v = v/np.linalg.norm(v)
    sims = norm @ v
    for i in ids: sims[i] = -1
    top = np.argsort(-sims)[:5]
    print(f"  {a.strip()} - {b.strip()} + {c.strip()} -> " +
          ", ".join(f"{tok.decode([int(j)]).strip()!r}" for j in top))

print("\n=== random baseline, for scale ===")
rng = np.random.default_rng(0)
r = rng.integers(0, E.shape[0], 4000)
vals = [float(norm[a] @ norm[b]) for a,b in zip(r[:2000], r[2000:])]
print(f"  mean similarity of 2,000 random token pairs: {np.mean(vals):+.4f}")
print(f"  standard deviation:                          {np.std(vals):.4f}")
