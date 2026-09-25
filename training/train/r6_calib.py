#!/usr/bin/env python3
"""R6 — quant calibration sensitivity. Re-quantize the extraction model with AWQ
using TASK-SPECIFIC calibration data (extraction prompts) instead of the default
generic corpus, and see if accuracy changes."""
import json, re, torch
from transformers import AutoTokenizer
from awq import AutoAWQForCausalLM

SRC = "models/hf/Qwen2.5-0.5B-extract-merged"; OUT = "models/hf/extract-awq-taskcal"
tok = AutoTokenizer.from_pretrained(SRC)
calib = [json.loads(l)["text"] for l in open("train/1_extract/data/train.jsonl")][:128]
print("quantizing with TASK calibration...", flush=True)
model = AutoAWQForCausalLM.from_pretrained(SRC)
model.quantize(tok, quant_config={"w_bit": 4, "q_group_size": 128, "zero_point": True, "version": "GEMM"},
               calib_data=calib)
model.save_quantized(OUT); tok.save_pretrained(OUT)
model = AutoAWQForCausalLM.from_quantized(OUT, fuse_layers=False)
tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
SYS = ("Extract the fields into JSON with exactly these keys: intent, time, amount, ref, person. "
 "intent is one of meeting, call, payment, reminder, deadline. time is 24-hour HH:MM or null. "
 "amount is a number or null. ref is a string or null. person is a string or null. Reply with only the JSON.")
KEYS = ["intent","time","amount","ref","person"]
def exj(s):
    m = re.search(r"\{.*\}", s, re.S)
    try: return json.loads(m.group(0)) if m else None
    except: return None
rows = [json.loads(l) for l in open("train/1_extract/data/hard.jsonl")]
prompts = [tok.apply_chat_template([{"role":"system","content":SYS},{"role":"user","content":r["text"]}],
           tokenize=False, add_generation_prompt=True) for r in rows]
preds = []
dev = next(model.model.parameters()).device
for i in range(0, len(prompts), 16):
    enc = tok(prompts[i:i+16], return_tensors="pt", padding=True).to(dev)
    with torch.no_grad():
        o = model.model.generate(**enc, max_new_tokens=80, do_sample=False, pad_token_id=tok.pad_token_id)
    preds += tok.batch_decode(o[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
ex = 0
for r, p in zip(rows, preds):
    o = exj(p) or {}; g = r["gold"]; ok = True
    for k in KEYS:
        gv, pv = g.get(k), o.get(k)
        if k == "amount" and isinstance(gv,(int,float)) and isinstance(pv,(int,float)): ok = ok and float(gv)==float(pv)
        else: ok = ok and gv == pv
    ex += ok
print(f"AWQ (TASK calibration) extraction EXACT: {ex/len(rows):.1%}  (vs 99.0% default calibration)")
print("=== R6 done ===")
