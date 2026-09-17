#!/usr/bin/env python3
"""Task 2 — execution accuracy for text -> SQL.

Score by RESULTS, not SQL text: run the predicted query and the gold query
against sql.db and compare the rows (order-insensitive, floats rounded). So a
differently-written query that returns the same rows counts as correct — the
fair metric for SQL, and the one that gives a general model its best shot.

Local:  --adapter train/2_sql/adapter        (transformers)
API:    --base-url http://10.0.0.20:8087/v1  (llama.cpp server or a frontier API)
"""
import json, argparse, re, sqlite3, time

def build_sys(schema_path):
    schema = open(schema_path).read()
    return ("You write SQLite queries against this schema:\n\n" + schema +
            "\n\nGiven a question, reply with only one SQL query, no explanation.")

def clean_sql(s):
    s = re.sub(r"```sql|```", "", s)
    m = re.search(r"(SELECT|WITH)\b.*", s, re.IGNORECASE | re.DOTALL)
    s = m.group(0) if m else s
    return s.split(";")[0].strip()

def norm(rows):
    out = []
    for row in rows:
        out.append(tuple(round(x, 2) if isinstance(x, float) else x for x in row))
    return sorted(out, key=lambda r: [str(x) for x in r])

def run(con, sql):
    try:
        return norm(con.execute(sql).fetchall()), True
    except Exception:
        return None, False

def gen_local(rows, base, adapter, sys, quant=None):
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
        [{"role":"system","content":sys},{"role":"user","content":r["q"]}],
        tokenize=False, add_generation_prompt=True) for r in rows]
    preds = []
    for i in range(0, len(prompts), 32):
        enc = tok(prompts[i:i+32], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=128, do_sample=False, pad_token_id=tok.pad_token_id)
        preds += tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return preds

def gen_api(rows, base_url, model, key, sys, conc):
    import requests
    from concurrent.futures import ThreadPoolExecutor
    url = base_url.rstrip("/") + "/chat/completions"
    hdr = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    def ask(q):
        body = {"model": model, "messages": [{"role":"system","content":sys},
                {"role":"user","content":q}], "temperature": 0, "max_tokens": 160}
        for _ in range(3):
            try:
                r = requests.post(url, headers=hdr, json=body, timeout=120); r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except Exception as e:
                last = str(e); time.sleep(1)
        return f"__ERR__ {last}"
    with ThreadPoolExecutor(max_workers=conc) as ex:
        return list(ex.map(ask, [r["q"] for r in rows]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="models/hf/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--model", default="local")
    ap.add_argument("--api-key", default="none")
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--data", default="train/2_sql/data/hard.jsonl")
    ap.add_argument("--db", default="train/2_sql/sql.db")
    ap.add_argument("--schema", default="train/2_sql/schema.sql")
    ap.add_argument("--tag", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--quant", default=None, choices=[None, "int8", "nf4"])
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data)]
    if args.limit: rows = rows[:args.limit]
    sys = build_sys(args.schema)

    t0 = time.time()
    if args.base_url:
        preds = gen_api(rows, args.base_url, args.model, args.api_key, sys, args.conc)
    else:
        preds = gen_local(rows, args.base, args.adapter, sys, args.quant)
    dt = time.time() - t0

    con = sqlite3.connect(args.db)
    runs = match = 0
    for r, p in zip(rows, preds):
        psql = clean_sql(p)
        gres, _ = run(con, r["sql"])
        pres, ok = run(con, psql)
        if ok: runs += 1
        if ok and pres == gres: match += 1

    n = len(rows)
    tag = args.tag or (f"adapter={args.adapter}" if args.adapter else args.model)
    print(f"\n=== {tag} — n={n}, {dt:.1f}s ===")
    print(f"sql_runs      : {runs/n:6.1%}   (query executes without error)")
    print(f"EXEC_MATCH    : {match/n:6.1%}   <- the headline (same rows as gold)")
    print("--- samples ---")
    for r, p in list(zip(rows, preds))[:3]:
        print("  q   :", r["q"])
        print("  gold:", r["sql"])
        print("  pred:", clean_sql(p)[:160])

if __name__ == "__main__":
    main()
