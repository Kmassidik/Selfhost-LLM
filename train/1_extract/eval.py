#!/usr/bin/env python3
"""Task 1 — score a model on the extraction val set.

Because the gold is exact, scoring is ==, not a judge:
  json_valid  — did it emit parseable JSON with the right keys
  exact       — all five fields equal gold (the number that matters)
  per-field   — accuracy on intent / time / amount / ref / person

Run on the base model, then with --adapter, and compare the two exact rates.
"""
import json, argparse, re, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

SYS = (
    "Extract the fields into JSON with exactly these keys: "
    "intent, time, amount, ref, person. "
    "intent is one of meeting, call, payment, reminder, deadline. "
    "time is 24-hour HH:MM or null. amount is a number or null. "
    "ref is a string or null. person is a string or null. "
    "Reply with only the JSON."
)
KEYS = ["intent", "time", "amount", "ref", "person"]

def extract_json(s):
    # grab the first {...} block
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="models/hf/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", default="train/1_extract/data/val.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--quant", default=None, choices=[None, "int8", "nf4"])
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data)]
    if args.limit:
        rows = rows[:args.limit]

    tok = AutoTokenizer.from_pretrained(args.base)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    from transformers import BitsAndBytesConfig
    kw = dict(device_map="cuda")
    if args.quant == "int8":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif args.quant == "nf4":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    else:
        kw["dtype"] = "bfloat16"
    model = AutoModelForCausalLM.from_pretrained(args.base, **kw)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    prompts = []
    for r in rows:
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": r["text"]}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    preds = []
    t0 = time.time()
    for i in range(0, len(prompts), args.bs):
        chunk = prompts[i:i+args.bs]
        enc = tok(chunk, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=80, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        preds += tok.batch_decode(gen, skip_special_tokens=True)
    dt = time.time() - t0

    valid = exact = 0
    field_ok = {k: 0 for k in KEYS}
    for r, p in zip(rows, preds):
        gold = r["gold"]
        obj = extract_json(p)
        if obj is not None and all(k in obj for k in KEYS):
            valid += 1
        obj = obj or {}
        hits = 0
        for k in KEYS:
            gv = gold.get(k)
            pv = obj.get(k, "__MISSING__")
            # numeric amount: accept int/float equality
            if k == "amount" and isinstance(gv, (int, float)) and isinstance(pv, (int, float)):
                ok = float(gv) == float(pv)
            else:
                ok = gv == pv
            if ok:
                field_ok[k] += 1
                hits += 1
        if hits == len(KEYS):
            exact += 1

    n = len(rows)
    tag = f"adapter={args.adapter}" if args.adapter else "BASE (no adapter)"
    print(f"\n=== {tag} — n={n}, {dt:.1f}s ({n/dt:.1f} ex/s) ===")
    print(f"json_valid : {valid/n:6.1%}")
    print(f"EXACT      : {exact/n:6.1%}   <- the headline")
    for k in KEYS:
        print(f"  {k:8s}: {field_ok[k]/n:6.1%}")
    # a couple of examples
    print("--- samples ---")
    for r, p in list(zip(rows, preds))[:3]:
        print("  in :", r["text"])
        print("  gold:", json.dumps(r["gold"]))
        print("  out :", p.strip().replace("\n", " ")[:160])

if __name__ == "__main__":
    main()
