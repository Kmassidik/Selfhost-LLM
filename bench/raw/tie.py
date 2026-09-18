import os, importlib.util, torch
HERE = "/root/Desktop/selfhostllm/source"
spec = importlib.util.spec_from_file_location("kv", HERE + "/15c_kv_cache.py")
kv = importlib.util.module_from_spec(spec); spec.loader.exec_module(kv)
fwd = kv.fwd
dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
W, cfg = fwd.load(dev)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(fwd.MODEL)
text = tok.apply_chat_template([{"role":"user","content":"The capital of France is"}],
                               tokenize=False, add_generation_prompt=True)
ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
with torch.no_grad():
    lg = kv.forward_cached(ids, W, cfg, dev, kv.KVCache(cfg["num_hidden_layers"]))[0, -1].float()

v = torch.topk(lg, 45).values
print("scores at ranks 38-43 (0-indexed):")
for r in range(37, 43):
    print(f"  rank {r:>2}  {v[r].item():.6f}")
kth = torch.topk(lg, 40).values[-1]
n = int((lg >= kth).sum())
ties = int((lg == kth).sum())
print(f"\n40th score = {kth.item():.6f}")
print(f"tokens with score >= that: {n}   tokens exactly equal to it: {ties}")
print(f"-> top_k=40 keeps {n} because {ties} tokens tie at the cut.")
print("\nbf16 has ~3 decimal digits, so ties at the boundary are common, not rare.")
