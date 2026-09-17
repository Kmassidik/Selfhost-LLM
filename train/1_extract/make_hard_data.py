#!/usr/bin/env python3
"""Task 1 — an ADVERSARIAL held-out set, from a different generator.

Same schema and gold rules as make_data.py, but built to defeat a model that
merely memorised the training templates:
  - a disjoint name pool and new lead/tail phrasings
  - light typos (a dropped or swapped char)
  - DISTRACTORS: a phone number and a room/desk number that look like refs or
    amounts but must stay out of the JSON (gold ignores them)
  - different field orderings

If the tuned model still scores high here, it learned the schema semantics
(money -> amount, invoice# -> ref), not the surface form. Written as hard.jsonl.
"""
import json, random, argparse
random.seed(101)

FIRST = ["Aiko","Bruno","Carmen","Dev","Esra","Farid","Gita","Hugo","Ines","Jonas",
         "Kito","Lucia","Milo","Noor","Otto","Pia","Quinn","Rosa","Silas","Uma"]
LAST  = ["Adler","Bianchi","Costa","Dubois","Eriksen","Faber","Gallo","Hoshino",
         "Iqbal","Jansen","Kraus","Lindqvist","Moreau","Nakamura","Oduya","Pires"]

def a_person():
    return f"{random.choice(FIRST)} {random.choice(LAST)}"

def a_time():
    h24 = random.randint(7, 20); minute = random.choice([0,0,0,15,30,45])
    gold = f"{h24:02d}:{minute:02d}"
    ap = "am" if h24 < 12 else "pm"; h12 = h24 if h24 <= 12 else h24-12
    forms = [f"{h12}.{minute:02d}{ap}", f"at {h24:02d}{minute:02d}h",
             f"{h12}:{minute:02d} in the {'morning' if h24<12 else 'evening'}", f"{h12}:{minute:02d}{ap}"]
    if minute == 0:  # minute-dropping forms are only unambiguous on the hour
        forms += [f"{h12} {ap}", f"{h12} o'clock {ap}", f"{h12}{ap} sharp"]
    return random.choice(forms), gold

def an_amount():
    val = random.choice([40, 60, 150, 250, 899, 1500, 3200, 7800, 9999, 15000, 130, 275])
    forms = [f"${val:,}", f"{val} bucks", f"USD{val}", f"€{val:,}", f"{val} euro", f"${val:,}.00"]
    return random.choice(forms), val

def a_ref():
    n = random.randint(1000, 99999)
    forms = [f"inv-{n}", f"po#{n}", f"ticket {n}", f"case {n}", f"#{n}"]
    return random.choice(forms), str(n)

def a_distractor():
    # things that look number-y but are NOT amount/ref/time
    kind = random.random()
    if kind < 0.4:
        return f"my cell is 555-0{random.randint(100,199)}"
    if kind < 0.7:
        return f"room {random.randint(2,40)}"
    return f"desk {random.randint(2,99)}"

INTENTS = ["meeting","call","payment","reminder","deadline"]

def typo(s):
    if len(s) < 4 or random.random() > 0.5:
        return s
    i = random.randint(1, len(s)-2)
    if random.random() < 0.5:              # drop a char
        return s[:i] + s[i+1:]
    return s[:i] + s[i+1] + s[i] + s[i+2:] # swap two chars

def make_row():
    intent = random.choice(INTENTS)
    gold = {"intent": intent, "time": None, "amount": None, "ref": None, "person": None}
    verb = {"meeting":"catch up","call":"phone","payment":"settle",
            "reminder":"remember to review","deadline":"ship"}[intent]
    # each fragment is (text, protected): protected fragments carry an answer and
    # are NEVER typo'd, so the gold stays readable from the text. Only filler is
    # noised.
    frags = [(verb, False)]

    if random.random() < 0.7:
        p = a_person(); gold["person"] = p
        frags.append((random.choice([f"with {p}", f"and loop in {p}", f"— {p} —"]), True))
    if random.random() < 0.75:
        tform, tgold = a_time(); gold["time"] = tgold
        frags.append((tform, True))
    if (intent in ("payment","deadline") and random.random() < 0.85) or random.random() < 0.2:
        aform, agold = an_amount(); gold["amount"] = agold
        frags.append((random.choice([f"totalling {aform}", f"— {aform}", f"worth {aform}"]), True))
    if (intent in ("payment","deadline") and random.random() < 0.8) or random.random() < 0.15:
        rform, rgold = a_ref(); gold["ref"] = rgold
        frags.append((random.choice([f"against {rform}", f"[{rform}]", f"see {rform}"]), True))
    # inject 0-2 distractors (protected so they stay recognisably number-like)
    for _ in range(random.randint(0, 2)):
        frags.insert(random.randint(1, len(frags)), (a_distractor(), True))

    head, tail = frags[0], frags[1:]
    random.shuffle(tail)                       # jumble everything after the verb
    frags = [head] + tail
    lead = random.choice(["", "urgent: ", "note — ", "heads up ", "when you can, ", "btw "])
    # typo only the filler words; keep protected fragments pristine
    out = [lead.strip()] if lead.strip() else []
    for txt, protected in frags:
        out.append(txt if protected else " ".join(typo(w) for w in txt.split()))
    return {"text": " ".join(" ".join(out).split()), "gold": gold}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", default="train/1_extract/data/hard.jsonl")
    args = ap.parse_args()
    seen, rows = set(), []
    while len(rows) < args.n:
        r = make_row()
        if r["text"] in seen: continue
        seen.add(r["text"]); rows.append(r)
    with open(args.out, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(f"hard: {len(rows)} rows -> {args.out}")
    for r in rows[:4]:
        print("  ", r["text"], "=>", json.dumps(r["gold"]))

if __name__ == "__main__":
    main()
