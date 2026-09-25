#!/usr/bin/env python3
"""R3 — produce answers for the 120 test questions under 4 configs:
base closed-book, base+RAG, tuned closed-book, tuned+RAG. Saves r3_answers.json."""
import json, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
sys.path.insert(0, "rag")
from search import Retriever

BASE = "models/hf/Qwen2.5-0.5B-Instruct"
SYS = "Answer the question about a self-hosted-LLM field guide with a short, direct answer."
tok = AutoTokenizer.from_pretrained(BASE); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
retr = Retriever()
test = [json.loads(l) for l in open("rag/data/qa_test.jsonl")]

def build(q, rag):
    ctx = ""
    if rag:
        hits = retr.search(q, 3)
        ctx = "Use this context:\n" + "\n".join(r["text"][:400] for _, r in hits) + "\n\n"
    return tok.apply_chat_template([{"role":"system","content":SYS},{"role":"user","content":ctx+q}],
                                   tokenize=False, add_generation_prompt=True)

def gen(model, prompts, mx=40):
    out = []
    for i in range(0, len(prompts), 16):
        enc = tok(prompts[i:i+16], return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda")
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=mx, do_sample=False, pad_token_id=tok.pad_token_id)
        out += [t.strip().replace("\n"," ") for t in tok.batch_decode(o[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
    return out

def load(adapter=None):
    m = AutoModelForCausalLM.from_pretrained(BASE, dtype="bfloat16", device_map="cuda")
    if adapter: m = PeftModel.from_pretrained(m, adapter)
    return m.eval()

base = load()
A = gen(base, [build(r["q"], False) for r in test])
B = gen(base, [build(r["q"], True) for r in test])
del base; torch.cuda.empty_cache()
tuned = load("rag/r3_adapter")
C = gen(tuned, [build(r["q"], False) for r in test])
D = gen(tuned, [build(r["q"], True) for r in test])

out = [{**test[i], "base_cb": A[i], "base_rag": B[i], "tuned_cb": C[i], "tuned_rag": D[i]} for i in range(len(test))]
json.dump(out, open("rag/data/r3_answers.json", "w"))
print(f"answered {len(out)} questions x4 configs -> rag/data/r3_answers.json")
for r in out[:2]:
    print("  Q:", r["q"][:50], "| gold:", r["a"][:30])
    print("    base_cb:", r["base_cb"][:40], "| base_rag:", r["base_rag"][:40])
    print("    tuned_cb:", r["tuned_cb"][:40], "| tuned_rag:", r["tuned_rag"][:40])
