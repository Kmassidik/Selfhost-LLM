"""Real data for the animation — CPU only, no GPU, so the benchmark is undisturbed."""
import json, struct
import numpy as np
M = "/root/Desktop/selfhostllm/models/hf/SmolLM2-360M-Instruct"
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(M)
cfg = json.load(open(M + "/config.json"))

PROMPT = "The capital of France is"
ids = tok(PROMPT, add_special_tokens=False).input_ids
pieces = [tok.decode([i]) for i in ids]

# read the embedding rows for those tokens straight out of the file, no torch
f = open(M + "/model.safetensors", "rb")
n = struct.unpack("<Q", f.read(8))[0]
hdr = json.loads(f.read(n)); start = 8 + n
spec = hdr["model.embed_tokens.weight"]
lo, _ = spec["data_offsets"]; D = spec["shape"][1]

def row(tid):
    f.seek(start + lo + tid * D * 2)
    raw = f.read(D * 2)
    u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
    return u16.view(np.float32) if False else np.frombuffer(u16.tobytes(), dtype=np.float32)

out = {
    "prompt": PROMPT,
    "config": {k: cfg[k] for k in ("hidden_size","num_hidden_layers","num_attention_heads",
                                   "num_key_value_heads","vocab_size","intermediate_size")},
    "tokens": [{"id": i, "text": p, "first8": [round(float(x), 4) for x in row(i)[:8]]}
               for i, p in zip(ids, pieces)],
}
f.close()
print(json.dumps(out, indent=1))
