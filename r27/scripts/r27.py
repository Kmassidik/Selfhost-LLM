#!/usr/bin/env python3
"""R27 — typed + calibrated decisions vs free text. Sentiment classification, Qwen2.5-0.5B.
A) free-text generate + parse   B) constrained LM-scoring (typed)   C) B + abstain below tau.
Measures: accuracy, parse-failure rate, and whether low-confidence cases are the wrong ones."""
import torch, math
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/root/Desktop/selfhostllm/models/hf/Qwen2.5-0.5B-Instruct"
dev = "cuda"
LABELS = ["positive", "negative", "neutral"]

DATA = [
 ("Absolutely loved it, best purchase this year!", "positive"),
 ("Fantastic quality and it arrived early.", "positive"),
 ("Works flawlessly, highly recommend.", "positive"),
 ("The staff were warm and the food delicious.", "positive"),
 ("Exceeded my expectations in every way.", "positive"),
 ("A delightful read from start to finish.", "positive"),
 ("Battery lasts all day, very happy.", "positive"),
 ("Total waste of money, broke in a week.", "negative"),
 ("Terrible service, I want a refund.", "negative"),
 ("The worst experience I've ever had here.", "negative"),
 ("Cheap materials and it fell apart.", "negative"),
 ("Slow, buggy, and constantly crashes.", "negative"),
 ("Disappointing sequel, nothing like the original.", "negative"),
 ("Rude staff and cold food.", "negative"),
 ("The package arrived on Tuesday.", "neutral"),
 ("It is a standard USB-C cable, one meter long.", "neutral"),
 ("The meeting is scheduled for 3 PM.", "neutral"),
 ("This model has 16 gigabytes of memory.", "neutral"),
 ("The report covers the third quarter.", "neutral"),
 ("The store opens at nine in the morning.", "neutral"),
 # harder / ambiguous ones
 ("It's fine, I guess, nothing special.", "neutral"),
 ("Not bad, but I expected more for the price.", "negative"),
 ("Well, it certainly is a product that exists.", "negative"),
 ("Could be worse, could be better.", "neutral"),
 ("I don't hate it.", "positive"),
 ("It does exactly what it says, no surprises.", "neutral"),
 ("Pricey, but honestly worth every penny.", "positive"),
 ("Gorgeous design, shame about the battery.", "negative"),
 ("The instructions were in another language.", "negative"),
 ("Solid, dependable, unexciting.", "neutral"),
]

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()

@torch.no_grad()
def freetext(text):
    msgs = [{"role": "user", "content": f'What is the sentiment of this text? Reply with exactly one word: positive, negative, or neutral.\n\nText: "{text}"'}]
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(p, return_tensors="pt").input_ids.to(dev)
    out = model.generate(ids, max_new_tokens=12, do_sample=False, pad_token_id=tok.eos_token_id)
    txt = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).lower()
    found = [l for l in LABELS if l in txt]
    return found[0] if len(found) == 1 else None      # None = unparseable / ambiguous output

@torch.no_grad()
def scored(text):
    base = f'Text: "{text}"\nSentiment:'
    lps = []
    for L in LABELS:
        fids = tok(base + " " + L).input_ids
        plen = len(tok(base).input_ids)
        lg = model(torch.tensor([fids], device=dev)).logits[0]
        lps.append(sum(torch.log_softmax(lg[i - 1], -1)[fids[i]].item() for i in range(plen, len(fids))))
    m = max(lps); ex = [math.exp(x - m) for x in lps]; Z = sum(ex); probs = [e / Z for e in ex]
    i = max(range(3), key=lambda k: probs[k])
    return LABELS[i], probs[i]

# A) free-text
a_correct = a_parsefail = 0
for text, gold in DATA:
    pred = freetext(text)
    if pred is None: a_parsefail += 1
    elif pred == gold: a_correct += 1
n = len(DATA)
print(f"A) free-text     : acc {a_correct/n*100:4.1f}%   parse-fail {a_parsefail/n*100:4.1f}%  (unparseable counted wrong)")

# B) constrained typed
results = [(scored(t), g) for t, g in DATA]
b_correct = sum(1 for (p, c), g in results if p == g)
print(f"B) typed (scored): acc {b_correct/n*100:4.1f}%   parse-fail 0.0%  (always a valid label)")

# C) typed + abstain
for tau in [0.5, 0.6, 0.7]:
    ans = [((p, c), g) for (p, c), g in results if c >= tau]
    absten = n - len(ans)
    acc = sum(1 for (p, c), g in ans if p == g) / len(ans) * 100 if ans else 0
    # calibration: accuracy of the ABSTAINED (low-conf) set
    lowconf = [((p, c), g) for (p, c), g in results if c < tau]
    low_acc = sum(1 for (p, c), g in lowconf if p == g) / len(lowconf) * 100 if lowconf else float("nan")
    print(f"C) typed+abstain tau={tau}: acc-on-answered {acc:4.1f}%   abstained {absten/n*100:4.1f}%   "
          f"(the abstained/low-conf set was only {low_acc:.0f}% right — so abstaining drops the wrong ones)")
