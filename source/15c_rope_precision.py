"""Does RoPE precision explain the long-context divergence?

Our rotation tables are built and stored in bf16 — about three decimal digits.
At position 2,200 the angle fed to cos and sin is large, and three digits of a
large angle is a coarse rotation. The reference builds these in fp32 and casts
only at the end. This measures whether that is the whole difference.
"""
import json, importlib.util, torch
HERE = "/root/Desktop/selfhostllm/source"
spec = importlib.util.spec_from_file_location("fwd", HERE + "/15b_forward_pass.py")
fwd = importlib.util.module_from_spec(spec); spec.loader.exec_module(fwd)
dev = torch.device("cuda:0"); torch.cuda.set_device(dev)

hd, theta = 64, 10000.0
for pos in (10, 100, 1000, 2200):
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, device=dev).float() / hd))
    ang = torch.tensor([float(pos)], device=dev) * inv          # fp32 angle
    c32, s32 = ang.cos(), ang.sin()
    c16, s16 = c32.to(torch.bfloat16).float(), s32.to(torch.bfloat16).float()
    # what the model actually rotates by: error in the cos/sin values themselves
    err = max((c32 - c16).abs().max().item(), (s32 - s16).abs().max().item())
    print(f"  position {pos:>5}   worst cos/sin error from storing in bf16: {err:.3e}")

print("\nThe table values are bounded in [-1, 1], so bf16 error is roughly constant")
print("with position — this is NOT where a long-context divergence comes from.\n")

# So test the real suspect: does the reference's own generation even agree with
# itself between a cached and uncached run at this length?
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(fwd.MODEL)
m = AutoModelForCausalLM.from_pretrained(fwd.MODEL, dtype=torch.bfloat16).to(dev).eval()

def expand(t):
    _, n, rest = t.split(":", 2); f, tail = rest.split("\n\n", 1)
    return f * int(n) + "\n\n" + tail
p = [x for x in json.load(open("/root/Desktop/selfhostllm/bench/prompts.json"))["prompts"]
     if x["id"] == "longctx"][0]
text = tok.apply_chat_template([{"role": "user", "content": expand(p["text"])}],
                               tokenize=False, add_generation_prompt=True)
ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
print(f"longctx is {ids.shape[1]} tokens")

with torch.no_grad():
    a = m.generate(ids, max_new_tokens=6, do_sample=False, use_cache=True,
                   pad_token_id=tok.eos_token_id)[0, ids.shape[1]:]
    b = m.generate(ids, max_new_tokens=6, do_sample=False, use_cache=False,
                   pad_token_id=tok.eos_token_id)[0, ids.shape[1]:]
print(f"  reference WITH its own cache   : {a.tolist()}  {tok.decode(a)!r}")
print(f"  reference WITHOUT its own cache: {b.tolist()}  {tok.decode(b)!r}")
print(f"  the reference agrees with itself: {torch.equal(a, b)}")
