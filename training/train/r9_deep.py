#!/usr/bin/env python3
"""R9 (deep) — the real multi-adapter serving bottleneck.

R8 showed 50 adapters FIT in one card's memory. This asks the harder question:
can you SERVE many different adapters at once? With peft only one adapter is
active at a time, so a mixed stream (requests to different adapters) cannot be
batched together the way a single-model stream can. Three regimes, same 96
requests, measured end to end:

  1. HOMOGENEOUS      — 96 requests all to one adapter; batches normally.
  2. MIXED, NAIVE     — 96 requests round-robin across 3 adapters, served in
                        arrival order; every request may switch adapter, so no
                        two adjacent requests batch together -> serialized.
  3. MIXED, GROUPED   — same 96, but sorted by adapter first: 3 batches, 2
                        switches. Adapter-aware scheduling restores batching.

The gap between (2) and (1)/(3) is the bottleneck: multi-adapter serving is a
SCHEDULER problem, not a memory problem. (3) is the poor-man's S-LoRA; true
S-LoRA/vLLM batches ACROSS adapters in one kernel — the next step.
"""
import json, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "models/hf/Qwen2.5-0.5B-Instruct"
PER = 32                      # requests per adapter
tok = AutoTokenizer.from_pretrained(BASE); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token

model = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained(BASE, dtype="bfloat16", device_map="cuda"),
    "train/1_extract/adapter", adapter_name="extract")
model.load_adapter("train/2_sql/adapter", adapter_name="sql")
model.load_adapter("train/3_cls/adapter", adapter_name="cls")
model.eval()

EX_SYS = ("Extract the fields into JSON with exactly these keys: intent, time, amount, ref, person. "
 "Reply with only the JSON.")
schema = open("train/2_sql/schema.sql").read()
SQL_SYS = "You write SQLite queries against this schema:\n\n"+schema+"\n\nReply with only one SQL query."
LABELS = ["billing_payment_failure","billing_refund","account_login","account_password_reset",
 "shipping_delay","shipping_lost","product_defect","product_return","technical_bug",
 "technical_outage","feature_request","general_inquiry"]
CLS_SYS = "Classify the support ticket into exactly one category:\n"+", ".join(LABELS)+".\nReply with only the category."

def load(task, sys, field, n):
    rows = [json.loads(l) for l in open(f"train/{task}/data/hard.jsonl")][:n]
    return [(adapter_of[task], tok.apply_chat_template(
        [{"role":"system","content":sys},{"role":"user","content":r[field]}],
        tokenize=False, add_generation_prompt=True)) for r in rows]

adapter_of = {"1_extract":"extract","2_sql":"sql","3_cls":"cls"}
reqs = (load("1_extract", EX_SYS, "text", PER)
      + load("2_sql", SQL_SYS, "q", PER)
      + load("3_cls", CLS_SYS, "text", PER))

def gen_batch(prompts, mx=64):
    enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        o = model.generate(**enc, max_new_tokens=mx, do_sample=False, pad_token_id=tok.pad_token_id)
    return int((o.shape[1]-enc["input_ids"].shape[1]) * o.shape[0])  # ~tokens generated

def run(name, schedule):
    torch.cuda.synchronize(); t0 = time.time(); toks = 0
    for adapter, group in schedule:
        model.set_adapter(adapter)
        toks += gen_batch([p for _, p in group])
    torch.cuda.synchronize(); dt = time.time() - t0
    n = sum(len(g) for _, g in schedule)
    print(f"{name:22s}: {dt:5.1f}s  {n/dt:6.1f} req/s  {toks/dt:7.0f} tok/s")
    return n/dt

# warmup
model.set_adapter("extract"); gen_batch([reqs[0][1]])

# 1. HOMOGENEOUS: 96 extract requests (batched in 32s)
ex_prompts = [("extract", p) for _, p in load("1_extract", EX_SYS, "text", PER*3)]
homo = [("extract", ex_prompts[i:i+32]) for i in range(0, len(ex_prompts), 32)]

# 2. MIXED NAIVE: round-robin, arrival order -> each request its own switch (batch of 1)
import itertools
mixed = [reqs[i] for i in range(len(reqs))]
rr = []
for k in range(PER):
    for base in (0, PER, 2*PER):
        rr.append(mixed[base+k])
naive = [(a, [(a, p)]) for a, p in rr]   # batch size 1 each, switches every step

# 3. MIXED GROUPED: sort by adapter, batch within -> 3 batches, 2 switches
grouped = []
for a in ("extract","sql","cls"):
    g = [(a, p) for aa, p in reqs if aa == a]
    for i in range(0, len(g), 32):
        grouped.append((a, g[i:i+32]))

print(f"=== R9 deep: multi-adapter serving, {PER*3} requests ===")
h = run("1 homogeneous", homo)
n = run("2 mixed, naive (FIFO)", naive)
g = run("3 mixed, grouped", grouped)
print(f"\nbottleneck: naive mixed serving is {h/n:.1f}x slower than homogeneous,")
print(f"adapter-aware grouping recovers it to {g/n:.1f}x faster than naive ({g/h*100:.0f}% of homogeneous).")
print("=== R9 done ===")
