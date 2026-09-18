"""The real exam for L4e: does our engine match the reference AT EACH BATCH SIZE?

Batch-size dependence is not ours — the reference has it too. So the standard
cannot be "one true output". It is: whatever the reference does at batch N,
we do at batch N.
"""
import json, hashlib, urllib.request, concurrent.futures as cf, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

URL = "http://10.0.0.20:8085/v1/chat/completions"
M = "/root/Desktop/selfhostllm/models/hf/SmolLM2-360M-Instruct"
PROMPTS = ["Hi", "The capital of France is",
           "Explain in one sentence what a KV cache is.",
           "Write a Python function that adds two numbers and returns the result."]
N = 40

def ask(p):
    b = json.dumps({"messages":[{"role":"user","content":p}],
                    "max_tokens":N,"temperature":0}).encode()
    r = urllib.request.Request(URL, b, {"Content-Type":"application/json"})
    with urllib.request.urlopen(r, timeout=300) as resp:
        return json.load(resp)["choices"][0]["message"]["content"]

ours_1 = [ask(p) for p in PROMPTS]
with cf.ThreadPoolExecutor(max_workers=4) as ex:
    ours_4 = list(ex.map(ask, PROMPTS))

dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
tok = AutoTokenizer.from_pretrained(M); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).to(dev).eval()
chat = [tok.apply_chat_template([{"role":"user","content":p}], tokenize=False,
                                add_generation_prompt=True) for p in PROMPTS]
with torch.no_grad():
    ref_1 = []
    for c in chat:
        i = tok(c, return_tensors="pt", add_special_tokens=False).to(dev)
        o = m.generate(**i, max_new_tokens=N, do_sample=False, pad_token_id=tok.pad_token_id)
        ref_1.append(tok.decode(o[0, i.input_ids.shape[1]:], skip_special_tokens=True))
    e = tok(chat, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
    o = m.generate(**e, max_new_tokens=N, do_sample=False, pad_token_id=tok.pad_token_id)
    ref_4 = [tok.decode(o[i, e.input_ids.shape[1]:], skip_special_tokens=True)
             for i in range(len(PROMPTS))]

print(f"{'prompt':<34}{'ours=ref @batch1':>18}{'ours=ref @batch4':>18}")
print("-" * 70)
a = b = 0
for p, o1, o4, r1, r4 in zip(PROMPTS, ours_1, ours_4, ref_1, ref_4):
    m1, m4 = o1.strip() == r1.strip(), o4.strip() == r4.strip()
    a += m1; b += m4
    print(f"{p[:32]!r:<34}{('MATCH' if m1 else 'differs'):>18}{('MATCH' if m4 else 'differs'):>18}")
print("-" * 70)
print(f"{'':<34}{f'{a}/4':>18}{f'{b}/4':>18}")
print()
print(f"batch changes the answer for us : {sum(1 for x,y in zip(ours_1,ours_4) if x!=y)} of 4")
print(f"batch changes it for the reference: {sum(1 for x,y in zip(ref_1,ref_4) if x!=y)} of 4")
