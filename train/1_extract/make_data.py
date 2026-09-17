#!/usr/bin/env python3
"""Task 1 — text -> JSON extraction. Synthetic, so the gold is exact.

A message like "call me 3pm re: invoice #4471, $2,300" must become

    {"intent":"call","time":"15:00","amount":2300,"ref":"4471","person":null}

Every field's correct answer is known by construction, so eval.py scores with
==, not a judge. This generator draws from a BROAD distribution on purpose:
  - many surface phrasings per intent (so intent isn't a single-verb lexical cue)
  - every time / currency / ref format
  - distractors (a phone or room number) that must stay out of the JSON
  - light typos on filler words (never on the answer tokens)
The held-out split is disjoint instances of this same rich distribution, so a
high val score means the schema was learned, not a template memorised.
Writes train.jsonl / val.jsonl as {"text":..., "gold": {...}}.
"""
import json, random, argparse

FIRST = ["Sam","Priya","Marcus","Lena","Diego","Amara","Tom","Yuki","Nadia","Carl",
         "Ravi","Elise","Omar","Grace","Hans","Mei","Ivan","Sofia","Ben","Zoe",
         "Aiko","Bruno","Carmen","Dev","Esra","Farid","Gita","Hugo","Ines","Jonas"]
LAST  = ["Ng","Patel","Reid","Cole","Vega","Osei","Frey","Sato","Khan","Diaz",
         "Roy","Blum","Aziz","Park","Voss","Lim","Petrov","Marin","Lowe","Tan",
         "Adler","Bianchi","Costa","Dubois","Eriksen","Faber","Gallo","Nakamura"]

def a_person():
    return f"{random.choice(FIRST)} {random.choice(LAST)}"

def a_time():
    h24 = random.randint(7, 20); minute = random.choice([0,0,0,15,30,45])
    gold = f"{h24:02d}:{minute:02d}"
    ap = "am" if h24 < 12 else "pm"; h12 = h24 if h24 <= 12 else h24-12
    forms = [f"{h12}:{minute:02d}{ap}", f"{h12}:{minute:02d} {ap.upper()}", f"{h24:02d}:{minute:02d}",
             f"{h12}.{minute:02d}{ap}", f"at {h24:02d}{minute:02d}h",
             f"{h12}:{minute:02d} in the {'morning' if h24<12 else 'evening'}"]
    if minute == 0:
        forms += [f"{h12}{ap}", f"{h12} {ap}", f"{h12} o'clock {ap}", f"{h12}{ap} sharp", f"{h12}:00 {ap}"]
    return random.choice(forms), gold

def an_amount():
    val = random.choice([25,40,60,75,90,120,150,199,250,275,300,450,899,1200,1500,
                         2300,3200,3400,4999,7800,9999,12000,15000,130])
    forms = [f"${val:,}", f"{val} dollars", f"${val}", f"USD {val:,}", f"{val:,} USD",
             f"${val:,}.00", f"{val} bucks", f"USD{val}", f"€{val:,}", f"{val} euro"]
    return random.choice(forms), val

def a_ref():
    n = random.randint(1000, 99999)
    forms = [f"#{n}", f"no. {n}", f"ref {n}", f"invoice {n}", f"order {n}",
             f"inv-{n}", f"po#{n}", f"ticket {n}", f"case {n}"]
    return random.choice(forms), str(n)

def a_distractor():
    r = random.random()
    if r < 0.4: return f"my cell is 555-0{random.randint(100,199)}"
    if r < 0.7: return f"room {random.randint(2,40)}"
    return f"desk {random.randint(2,99)}"

# several surface phrasings per intent, so intent is a concept, not one word
VERBS = {
    "meeting":  ["meet", "catch up", "sync up", "get together", "have a meeting"],
    "call":     ["call", "phone", "ring", "give a call", "hop on a call"],
    "payment":  ["pay", "settle", "send payment", "wire", "make the payment"],
    "reminder": ["remind me to sync", "remember to review", "don't let me forget", "note to self:", "remind me about"],
    "deadline": ["deliver", "ship", "submit", "finish", "have ready"],
}
INTENTS = list(VERBS)

def typo(s):
    if len(s) < 4 or random.random() > 0.25:
        return s
    i = random.randint(1, len(s)-2)
    if random.random() < 0.5:
        return s[:i] + s[i+1:]
    return s[:i] + s[i+1] + s[i] + s[i+2:]

def make_row():
    intent = random.choice(INTENTS)
    gold = {"intent": intent, "time": None, "amount": None, "ref": None, "person": None}
    frags = [(random.choice(VERBS[intent]), False)]

    if random.random() < 0.7:
        p = a_person(); gold["person"] = p
        frags.append((random.choice([f"with {p}", f"{p}", f"w/ {p}", f"and loop in {p}"]), True))
    if random.random() < 0.75:
        tf, tg = a_time(); gold["time"] = tg
        frags.append((random.choice([f"at {tf}", f"{tf}", f"@ {tf}"]), True))
    if (intent in ("payment","deadline") and random.random() < 0.85) or random.random() < 0.2:
        af, ag = an_amount(); gold["amount"] = ag
        frags.append((random.choice([f"for {af}", f"{af}", f"totalling {af}", f"worth {af}"]), True))
    if (intent in ("payment","deadline") and random.random() < 0.8) or random.random() < 0.15:
        rf, rg = a_ref(); gold["ref"] = rg
        frags.append((random.choice([f"re: {rf}", f"({rf})", f"{rf}", f"against {rf}"]), True))
    for _ in range(random.randint(0, 2)):
        frags.insert(random.randint(1, len(frags)), (a_distractor(), True))

    head, tail = frags[0], frags[1:]
    random.shuffle(tail)
    lead = random.choice(["", "hey ", "quick one — ", "pls ", "fyi ", "urgent: ",
                          "note — ", "heads up ", "when you can, ", "btw ", "don't forget: "])
    parts = [lead.strip()] if lead.strip() else []
    for txt, protected in [head] + tail:
        parts.append(txt if protected else " ".join(typo(w) for w in txt.split()))
    tail_s = random.choice(["", " thanks", " ok?", " cheers", " asap", " pls", ""])
    text = " ".join(" ".join(parts).split()) + tail_s
    return {"text": " ".join(text.split()), "gold": gold}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=3000)
    ap.add_argument("--val", type=int, default=400)
    ap.add_argument("--out", default=".")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    random.seed(args.seed)

    seen = set()
    def batch(n):
        rows = []
        while len(rows) < n:
            r = make_row()
            if r["text"] in seen: continue
            seen.add(r["text"]); rows.append(r)
        return rows

    for name, n in [("train", args.train), ("val", args.val)]:
        rows = batch(n)
        with open(f"{args.out}/{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{name}: {len(rows)} rows -> {args.out}/{name}.jsonl")
        for r in rows[:3]:
            print("  ", r["text"], "=>", json.dumps(r["gold"]))

if __name__ == "__main__":
    main()
