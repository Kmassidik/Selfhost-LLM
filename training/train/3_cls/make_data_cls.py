#!/usr/bin/env python3
"""Task 3 — support-ticket routing. Text -> one of 12 fixed categories.

Synthetic: each category has train phrasings and a disjoint pool of hard
phrasings (--hard) the model never trains on, so the hard set tests whether the
category concept generalises past the training templates. Scored by accuracy and
macro-F1. Rows: {"text":..., "label": category}.
"""
import json, random, argparse

LABELS = [
    "billing_payment_failure", "billing_refund", "account_login",
    "account_password_reset", "shipping_delay", "shipping_lost",
    "product_defect", "product_return", "technical_bug",
    "technical_outage", "feature_request", "general_inquiry",
]

PROD = ["the router","my tablet","the blender","the desk lamp","the headset","the kettle","the monitor"]

def s(t):  # fill {p} with a random product where present
    return t.replace("{p}", random.choice(PROD))

TRAIN = {
  "billing_payment_failure": ["my card was declined at checkout twice","payment keeps failing when I try to pay","the charge won't go through, card error","I can't complete payment, it errors out"],
  "billing_refund": ["I want a refund for my last order","please refund me, I was charged wrongly","how do I get my money back for this","requesting a refund on the duplicate charge"],
  "account_login": ["I can't log into my account","login isn't working for me","it says my credentials are wrong when I sign in","unable to access my account, login fails"],
  "account_password_reset": ["I need to reset my password","forgot my password, how do I change it","send me a password reset link please","how do I set a new password"],
  "shipping_delay": ["my order is late, where is it","the package hasn't arrived and it's overdue","shipping is taking way longer than promised","my delivery is delayed by a week"],
  "shipping_lost": ["my parcel never arrived, tracking is stuck","the courier lost my package","order marked delivered but I never got it","my shipment seems to be lost"],
  "product_defect": ["{p} arrived broken","{p} stopped working after a day","{p} is defective out of the box","there's a crack in {p}, it's faulty"],
  "product_return": ["I want to return {p}","how do I send {p} back","{p} doesn't fit, I'd like to return it","need to return {p} for a different size"],
  "technical_bug": ["the app crashes when I open settings","there's a bug in the checkout page","the button does nothing when I click it","the site throws an error on submit"],
  "technical_outage": ["the whole site is down right now","your service is unavailable, nothing loads","everything is offline, is there an outage","the API is returning 503 for everyone"],
  "feature_request": ["could you add a dark mode","it would be great to have export to CSV","please consider adding two-factor auth","I'd love a mobile app version"],
  "general_inquiry": ["what are your opening hours","do you ship internationally","where are you located","can you tell me more about your company"],
}
HARD = {
  "billing_payment_failure": ["transaction gets rejected every time I check out","the system won't accept my payment method","my visa bounces at the pay step"],
  "billing_refund": ["kindly reverse the charge on my account","I'd like my payment returned","can you issue money back for order 4471"],
  "account_login": ["sign-in refuses my details","stuck on the login screen, won't let me in","can't get past authentication"],
  "account_password_reset": ["help me change my passcode","I locked myself out and need a new password","the reset email never came, how do I set a password"],
  "shipping_delay": ["still waiting on a parcel that's overdue","when will my slow delivery actually show up","my order's been in transit far too long"],
  "shipping_lost": ["tracking says delivered yet nothing came","the box vanished in transit","carrier can't locate my consignment"],
  "product_defect": ["{p} came damaged","{p} is malfunctioning right away","the unit I received is faulty"],
  "product_return": ["I need to send {p} back for a refund","initiate a return for {p} please","how to ship {p} back to you"],
  "technical_bug": ["the page freezes midway through","clicking save produces a glitch","something's broken in the dashboard"],
  "technical_outage": ["nothing on your platform responds at all","total blackout, the app is dead","servers appear to be down globally"],
  "feature_request": ["any chance of a scheduling option","would be nice to bulk-edit entries","suggestion: support keyboard shortcuts"],
  "general_inquiry": ["are you open on weekends","which countries do you deliver to","just a general question about your team"],
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="train/3_cls/data")
    ap.add_argument("--train", type=int, default=1800)
    ap.add_argument("--val", type=int, default=300)
    ap.add_argument("--hard", type=int, default=300)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    import os; os.makedirs(args.out, exist_ok=True)
    random.seed(args.seed)

    def batch(n, pools):
        rows = []
        for i in range(n):
            lab = LABELS[i % len(LABELS)]  # balanced across classes
            rows.append({"text": s(random.choice(pools[lab])), "label": lab})
        random.shuffle(rows)
        return rows

    for name, n, pools in [("train", args.train, TRAIN), ("val", args.val, TRAIN), ("hard", args.hard, HARD)]:
        rows = batch(n, pools)
        with open(f"{args.out}/{name}.jsonl", "w") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
        print(f"{name}: {len(rows)} rows")
        for r in rows[:3]:
            print("   ", r["text"], "=>", r["label"])

if __name__ == "__main__":
    main()
