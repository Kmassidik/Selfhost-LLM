#!/usr/bin/env python3
"""R8 — multi-adapter serving (the mini-ixCSP core).

One base 0.5B held in memory once, with all three task adapters attached and
routed per request via set_adapter(). Measures:
  - VRAM(base + 3 adapters) vs VRAM(3 separate full models)  -> density
  - adapter switch latency
  - retained accuracy per task (should match the single-adapter baselines)
  - a density projection to N models on one card.
Runs on 100 hard-set rows per task for speed.
"""
import json, time, re, sqlite3, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "models/hf/Qwen2.5-0.5B-Instruct"
N = 100
tok = AutoTokenizer.from_pretrained(BASE); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token

def vram():
    torch.cuda.synchronize(); return torch.cuda.memory_allocated() / 1e9

base = AutoModelForCausalLM.from_pretrained(BASE, dtype="bfloat16", device_map="cuda")
v_base = vram()
t0 = time.time()
model = PeftModel.from_pretrained(base, "train/1_extract/adapter", adapter_name="extract")
model.load_adapter("train/2_sql/adapter", adapter_name="sql")
model.load_adapter("train/3_cls/adapter", adapter_name="cls")
model.eval()
v_all = vram()
print(f"base VRAM {v_base:.3f} GB; base+3 adapters {v_all:.3f} GB "
      f"(+{(v_all-v_base)*1000:.0f} MB for 3 adapters); load {time.time()-t0:.1f}s")

def gen(prompts, mx):
    out = []
    for i in range(0, len(prompts), 32):
        enc = tok(prompts[i:i+32], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=mx, do_sample=False, pad_token_id=tok.pad_token_id)
        out += tok.batch_decode(o[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return out

def prompts_for(rows, sys, field):
    return [tok.apply_chat_template(
        [{"role":"system","content":sys},{"role":"user","content":r[field]}],
        tokenize=False, add_generation_prompt=True) for r in rows]

# switch latency
lat = []
for a in ["extract","sql","cls"] * 4:
    t = time.time(); model.set_adapter(a); torch.cuda.synchronize(); lat.append((time.time()-t)*1000)
print(f"adapter switch latency: mean {sum(lat)/len(lat):.2f} ms, max {max(lat):.2f} ms")

# --- extract ---
EX_SYS = ("Extract the fields into JSON with exactly these keys: intent, time, amount, ref, person. "
 "intent is one of meeting, call, payment, reminder, deadline. time is 24-hour HH:MM or null. "
 "amount is a number or null. ref is a string or null. person is a string or null. Reply with only the JSON.")
KEYS = ["intent","time","amount","ref","person"]
def exj(s):
    m = re.search(r"\{.*\}", s, re.S)
    try: return json.loads(m.group(0)) if m else None
    except: return None
rows = [json.loads(l) for l in open("train/1_extract/data/hard.jsonl")][:N]
model.set_adapter("extract")
preds = gen(prompts_for(rows, EX_SYS, "text"), 80)
ex = 0
for r, p in zip(rows, preds):
    o = exj(p) or {}; g = r["gold"]; ok = True
    for k in KEYS:
        gv, pv = g.get(k), o.get(k)
        if k == "amount" and isinstance(gv,(int,float)) and isinstance(pv,(int,float)):
            if float(gv) != float(pv): ok = False
        elif gv != pv: ok = False
    ex += ok
print(f"extract (routed): EXACT {ex/len(rows):.1%}")

# --- sql ---
schema = open("train/2_sql/schema.sql").read()
SQL_SYS = "You write SQLite queries against this schema:\n\n"+schema+"\n\nGiven a question, reply with only one SQL query, no explanation."
def clean(s):
    s = re.sub(r"```sql|```","",s); m = re.search(r"(SELECT|WITH)\b.*", s, re.I|re.S)
    s = m.group(0) if m else s; return s.split(";")[0].strip()
def norm(rs): return sorted(tuple(round(x,2) if isinstance(x,float) else x for x in row) for row in rs)
con = sqlite3.connect("train/2_sql/sql.db")
def run(sql):
    try: return norm(con.execute(sql).fetchall())
    except: return None
rows = [json.loads(l) for l in open("train/2_sql/data/hard.jsonl")][:N]
model.set_adapter("sql")
preds = gen(prompts_for(rows, SQL_SYS, "q"), 128)
mt = sum(1 for r,p in zip(rows,preds) if run(clean(p)) is not None and run(clean(p)) == run(r["sql"]))
print(f"sql (routed): EXEC_MATCH {mt/len(rows):.1%}")

# --- cls ---
LABELS = ["billing_payment_failure","billing_refund","account_login","account_password_reset",
 "shipping_delay","shipping_lost","product_defect","product_return","technical_bug",
 "technical_outage","feature_request","general_inquiry"]
CLS_SYS = "Classify the support ticket into exactly one category:\n"+", ".join(LABELS)+".\nReply with only the category."
def plabel(t):
    t = t.strip().lower()
    for l in LABELS:
        if t == l or t.startswith(l): return l
    for l in LABELS:
        if l in t: return l
    return "__none__"
rows = [json.loads(l) for l in open("train/3_cls/data/hard.jsonl")][:N]
model.set_adapter("cls")
preds = gen(prompts_for(rows, CLS_SYS, "text"), 16)
golds = [r["label"] for r in rows]; ps = [plabel(x) for x in preds]
f1s = []
for l in LABELS:
    tp = sum(1 for g,p in zip(golds,ps) if g==l and p==l)
    fp = sum(1 for g,p in zip(golds,ps) if g!=l and p==l)
    fn = sum(1 for g,p in zip(golds,ps) if g==l and p!=l)
    pr = tp/(tp+fp) if tp+fp else 0; rc = tp/(tp+fn) if tp+fn else 0
    f1s.append(2*pr*rc/(pr+rc) if pr+rc else 0)
print(f"cls (routed): macro-F1 {sum(f1s)/len(f1s):.1%}")

# --- density projection ---
amb = (v_all - v_base) * 1000 / 3
print(f"\nDENSITY: base {v_base:.2f} GB + ~{amb:.0f} MB/adapter")
for n in [3,10,20,50]:
    multi = v_base + amb*n/1000; sep = v_base*n
    print(f"  {n:>2} models: multi-adapter {multi:.2f} GB  vs  {sep:.1f} GB separate  ({sep/multi:.1f}x denser)")
print("=== R8 done ===")
