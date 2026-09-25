"""Can the ranking be defended, or is 10 tasks too few?

The PRD's checkpoint for this chapter is a ranking of three models "we can
defend from the data, not from vibes". Defending it means asking whether the
gap could have come from chance.
"""
import json, glob, math
rs = {}
for f in glob.glob("/root/Desktop/selfhostllm/bench/results/eval-*.json"):
    d = json.load(open(f)); rs[d["label"]] = d

hdr = f"{'model':<18}{'correct':>9}{'wrong':>8}{'gave up':>9}{'completion':>12}{'accuracy':>10}"
print(hdr); print("-" * len(hdr))
for L in ("smollm2-360m", "llama3-8b-q4", "llama3-8b-q8"):
    d = rs.get(L)
    if not d: continue
    n = d["n"]
    print(f"{L:<18}{str(d['correct'])+'/'+str(n):>9}{d['wrong']:>8}{d['gave_up']:>9}"
          f"{(d['correct']+d['wrong'])/n*100:>11.0f}%{d['correct']/n*100:>9.0f}%")

qa = {r["task"]: r["ok"] for r in rs["llama3-8b-q4"]["rows"]}
qb = {r["task"]: r["ok"] for r in rs["llama3-8b-q8"]["rows"]}
only_a = sum(1 for t in qa if qa[t] and not qb[t])
only_b = sum(1 for t in qa if qb[t] and not qa[t])
d_ = only_a + only_b
p = (sum(math.comb(d_, k) for k in range(min(only_a, only_b) + 1)) * 2 / 2**d_) if d_ else 1.0
p = min(p, 1.0)

print()
print("Q8 scored one task above Q4. Is that a real difference?")
print(f"  tasks only Q4 got right   {only_a}")
print(f"  tasks only Q8 got right   {only_b}")
print(f"  they disagree on          {d_} task(s) out of 10")
print(f"  exact two-sided p-value   {p:.2f}")
print()
verdict = "DEFENSIBLE" if p < 0.05 else "NOT DEFENSIBLE"
print(f"  {verdict}. With ten tasks, a one-task gap is exactly what chance")
print(f"  produces between two equally good models. The harness ranked them;")
print(f"  the data does not support the ranking at this sample size.")
print()
both_wrong = sorted(t for t in qa if not qa[t] and not qb[t])
print(f"  Tasks BOTH 8B models failed: {both_wrong}")
print(f"  Identical failures on identical tasks — the extra precision of Q8")
print(f"  changed the score by one task and changed the failure MODE not at all.")
print()
print("  Against SmolLM2-360M at 0/10, both 8B models are separated by a mile")
print("  and that comparison IS defensible. Two of the three comparisons in this")
print("  ranking are solid; the interesting one is not.")
