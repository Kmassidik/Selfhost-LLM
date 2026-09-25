#!/usr/bin/env python3
"""Pull real scalar / vector / matrix / 3-D slices out of a model file.

Chapter 02. The numbers in that chapter's scalar-vector-matrix-tensor diagram
come from here, so the diagram can be checked rather than trusted.

    python3 source/02b_tensor_slices.py

Note the 3-D case has to be CONSTRUCTED by stacking two layers' matrices —
nothing in the file has three directions to begin with.
"""
import json, struct, os, numpy as np
D = os.path.join(os.environ.get("SELFHOSTLLM_ROOT","."), "models/hf/SmolLM2-360M-Instruct")
with open(f"{D}/model.safetensors","rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(n)); base = 8+n
    def get(name, rows, cols=None):
        m = hdr[name]; s,e = m["data_offsets"]
        f.seek(base+s); raw = f.read(e-s)
        a = (np.frombuffer(raw,dtype="<u2").astype(np.uint32)<<16).view(np.float32)
        a = a.reshape(m["shape"])
        return a[:rows,:cols] if cols else a[:rows]

    v = get("model.layers.0.input_layernorm.weight", 4)
    m = get("model.layers.0.self_attn.q_proj.weight", 4, 4)
    c = np.stack([get("model.layers.0.self_attn.q_proj.weight",2,3),
                  get("model.layers.1.self_attn.q_proj.weight",2,3)])

print("scalar  (one number from the grid):")
print(f"  {m[0,0]:.3f}")
print("\nvector  (first 4 of input_layernorm.weight, shape [960]):")
print("  [" + ", ".join(f"{x:.3f}" for x in v) + "]")
print("\nmatrix  (top-left 4x4 of q_proj.weight, shape [960, 960]):")
for r in m: print("  [" + ", ".join(f"{x:6.3f}" for x in r) + "]")
print("\n3D  (2 layers x 2 rows x 3 cols — stacking two matrices):")
for i,layer in enumerate(c):
    print(f"  layer {i}:")
    for r in layer: print("    [" + ", ".join(f"{x:6.3f}" for x in r) + "]")
import json, struct, os, numpy as np
D = os.path.join(os.environ.get("SELFHOSTLLM_ROOT","."), "models/hf/SmolLM2-360M-Instruct")
with open(f"{D}/model.safetensors","rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(n)); base = 8+n
    def get(name, rows, cols=None):
        m = hdr[name]; s,e = m["data_offsets"]
        f.seek(base+s); raw = f.read(e-s)
        a = (np.frombuffer(raw,dtype="<u2").astype(np.uint32)<<16).view(np.float32)
        a = a.reshape(m["shape"])
        return a[:rows,:cols] if cols else a[:rows]

    v = get("model.layers.0.input_layernorm.weight", 4)
    m = get("model.layers.0.self_attn.q_proj.weight", 4, 4)
    c = np.stack([get("model.layers.0.self_attn.q_proj.weight",2,3),
                  get("model.layers.1.self_attn.q_proj.weight",2,3)])

print("scalar  (one number from the grid):")
print(f"  {m[0,0]:.3f}")
print("\nvector  (first 4 of input_layernorm.weight, shape [960]):")
print("  [" + ", ".join(f"{x:.3f}" for x in v) + "]")
print("\nmatrix  (top-left 4x4 of q_proj.weight, shape [960, 960]):")
for r in m: print("  [" + ", ".join(f"{x:6.3f}" for x in r) + "]")
print("\n3D  (2 layers x 2 rows x 3 cols — stacking two matrices):")
for i,layer in enumerate(c):
    print(f"  layer {i}:")
    for r in layer: print("    [" + ", ".join(f"{x:6.3f}" for x in r) + "]")
