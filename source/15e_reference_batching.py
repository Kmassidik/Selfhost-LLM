"""Does the REFERENCE also change its answer when batched?

If batching shifts our logits because a batched matmul accumulates in a
different order, the same must happen to transformers. If it does, batch-size
dependence is a property of the hardware and the arithmetic, not of our engine.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
M = "/root/Desktop/selfhostllm/models/hf/SmolLM2-360M-Instruct"
dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
tok = AutoTokenizer.from_pretrained(M)
tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).to(dev).eval()

PROMPTS = ["Hi", "The capital of France is",
           "Explain in one sentence what a KV cache is.",
           "Write a Python function that adds two numbers and returns the result."]
chat = [tok.apply_chat_template([{"role":"user","content":p}], tokenize=False,
                                add_generation_prompt=True) for p in PROMPTS]

with torch.no_grad():
    alone = []
    for c in chat:
        i = tok(c, return_tensors="pt", add_special_tokens=False).to(dev)
        o = m.generate(**i, max_new_tokens=40, do_sample=False,
                       pad_token_id=tok.pad_token_id)
        alone.append(tok.decode(o[0, i.input_ids.shape[1]:], skip_special_tokens=True))

    enc = tok(chat, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
    o = m.generate(**enc, max_new_tokens=40, do_sample=False,
                   pad_token_id=tok.pad_token_id)
    batched = [tok.decode(o[i, enc.input_ids.shape[1]:], skip_special_tokens=True)
               for i in range(len(PROMPTS))]

print("THE REFERENCE, batch of 1 vs batch of 4, temperature 0, same weights")
print("-" * 70)
same_n = 0
for p, a, b in zip(PROMPTS, alone, batched):
    s = a == b
    same_n += s
    print(f"  {'SAME' if s else 'DIFFERS'}  {p[:44]!r}")
    if not s:
        print(f"      alone   {a[:88]!r}")
        print(f"      batched {b[:88]!r}")
print("-" * 70)
print(f"  the reference agrees with itself on {same_n} of {len(PROMPTS)} when batched")
