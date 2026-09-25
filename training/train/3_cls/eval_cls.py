#!/usr/bin/env python3
"""Task 3 — accuracy and macro-F1 for ticket routing.

Local:  --adapter train/3_cls/adapter
API:    --base-url http://10.0.0.20:8087/v1
"""
import json, argparse, time
from make_data_cls import LABELS

SYS = ("Classify the support ticket into exactly one category:\n" +
       ", ".join(LABELS) + ".\nReply with only the category.")

def parse_label(text):
    t = text.strip().lower()
    for lab in LABELS:                    # exact-ish first
        if t == lab or t.startswith(lab):
            return lab
    for lab in LABELS:                    # otherwise first label mentioned
        if lab in t:
            return lab
    return "__none__"

def macro_f1(golds, preds):
    f1s = []
    for lab in LABELS:
        tp = sum(1 for g, p in zip(golds, preds) if g == lab and p == lab)
        fp = sum(1 for g, p in zip(golds, preds) if g != lab and p == lab)
        fn = sum(1 for g, p in zip(golds, preds) if g == lab and p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s)

def gen_local(rows, base, adapter, quant=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(base); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    kw = dict(device_map="cuda")
    if quant == "int8":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quant == "nf4":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    else:
        kw["dtype"] = "bfloat16"
    model = AutoModelForCausalLM.from_pretrained(base, **kw)
    if adapter: model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    prompts = [tok.apply_chat_template(
        [{"role":"system","content":SYS},{"role":"user","content":r["text"]}],
        tokenize=False, add_generation_prompt=True) for r in rows]
    preds = []
    for i in range(0, len(prompts), 32):
        enc = tok(prompts[i:i+32], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=16, do_sample=False, pad_token_id=tok.pad_token_id)
        preds += tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return preds

def gen_api(rows, base_url, model, key, conc):
    import requests
    from concurrent.futures import ThreadPoolExecutor
    url = base_url.rstrip("/") + "/chat/completions"
    hdr = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    def ask(text):
        body = {"model": model, "messages": [{"role":"system","content":SYS},
                {"role":"user","content":text}], "temperature": 0, "max_tokens": 16}
        for _ in range(3):
            try:
                r = requests.post(url, headers=hdr, json=body, timeout=120); r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except Exception as e:
                last = str(e); time.sleep(1)
        return f"__ERR__ {last}"
    with ThreadPoolExecutor(max_workers=conc) as ex:
        return list(ex.map(ask, [r["text"] for r in rows]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="models/hf/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--model", default="local")
    ap.add_argument("--api-key", default="none")
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--data", default="train/3_cls/data/hard.jsonl")
    ap.add_argument("--tag", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--quant", default=None, choices=[None, "int8", "nf4"])
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data)]
    if args.limit: rows = rows[:args.limit]

    t0 = time.time()
    if args.base_url:
        raw = gen_api(rows, args.base_url, args.model, args.api_key, args.conc)
    else:
        raw = gen_local(rows, args.base, args.adapter, args.quant)
    dt = time.time() - t0

    golds = [r["label"] for r in rows]
    preds = [parse_label(x) for x in raw]
    acc = sum(1 for g, p in zip(golds, preds) if g == p) / len(rows)
    f1 = macro_f1(golds, preds)
    tag = args.tag or (f"adapter={args.adapter}" if args.adapter else args.model)
    print(f"\n=== {tag} — n={len(rows)}, {dt:.1f}s ===")
    print(f"accuracy   : {acc:6.1%}")
    print(f"MACRO_F1   : {f1:6.1%}   <- the headline")
    print("--- samples ---")
    for r, p in list(zip(rows, preds))[:3]:
        mark = "ok" if p == r["label"] else "X"
        print(f"  [{mark}] {r['text'][:60]:60s} gold={r['label']:24s} pred={p}")

if __name__ == "__main__":
    main()
