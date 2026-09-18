#!/usr/bin/env python3
"""
22_eval.py — the instrument (ch.22).

Chapter 21 showed a model completing every task and getting most of them wrong.
Nothing measured there could tell the two apart. This is the thing that can.

DESIGN DECISIONS, EACH ONE LOAD-BEARING:

  Ground truth is COMPUTED, never stored. Every task carries a function that
  works out the right answer from the repository at the moment the test runs.
  A stored answer key rots the first time a file is added; a computed one
  cannot be wrong about the repository it is reading.

  Tasks come from OUR repository, which no public model has seen. A benchmark
  made of public questions cannot distinguish a model that can reason from one
  that memorised the answer during training — contamination, and it is why
  published scores are so hard to trust.

  Grading is exact and mechanical. No model judges another model. The answer is
  a number or a yes, and it either matches what the filesystem says or it does not.

    uv run source/22_eval.py --endpoint URL --model NAME --label NAME
"""
import argparse, json, os, re, subprocess, sys, time, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/root/Desktop/selfhostllm"
_s = importlib.util.spec_from_file_location("ag", os.path.join(HERE, "21_agent.py"))
ag = importlib.util.module_from_spec(_s); _s.loader.exec_module(ag)


# ── ground truth, computed from the repository, every run ──────────────────
def n_files(d):      return str(len(os.listdir(os.path.join(ROOT, d))))
def n_lines(f):      return str(sum(1 for _ in open(os.path.join(ROOT, f))))
def exists(f):       return "yes" if os.path.exists(os.path.join(ROOT, f)) else "no"
def n_prompts():     return str(len(json.load(open(f"{ROOT}/bench/prompts.json"))["prompts"]))
def n_py(d):         return str(len([f for f in os.listdir(os.path.join(ROOT, d))
                                     if f.endswith(".py")]))
def cfg(key):        return str(json.load(open(
                         f"{ROOT}/models/hf/SmolLM2-360M-Instruct/config.json"))[key])
def n_grep(pat, d):
    out = subprocess.run(["grep", "-rl", pat, os.path.join(ROOT, d)],
                         capture_output=True, text=True).stdout
    return str(len([x for x in out.split("\n") if x.strip()]))


TASKS = [
    ("files_source",  "How many files are in the source directory? Answer with only the number.",
     lambda: n_files("source")),
    ("py_source",     "How many files in the source directory end with .py? Answer with only the number.",
     lambda: n_py("source")),
    ("run_exists",    "Does the bench directory contain a file called run.py? Answer yes or no.",
     lambda: exists("bench/run.py")),
    ("ghost_exists",  "Does the bench directory contain a file called trainer.py? Answer yes or no.",
     lambda: exists("bench/trainer.py")),
    ("n_prompts",     "The file bench/prompts.json has a list called prompts. How many entries "
                      "are in that list? Answer with only the number.",
     lambda: n_prompts()),
    ("readme_lines",  "How many lines are in README.md? Answer with only the number.",
     lambda: n_lines("README.md")),
    ("layers",        "The file models/hf/SmolLM2-360M-Instruct/config.json has a field "
                      "num_hidden_layers. What is its value? Answer with only the number.",
     lambda: cfg("num_hidden_layers")),
    ("kv_heads",      "In models/hf/SmolLM2-360M-Instruct/config.json, what is the value of "
                      "num_key_value_heads? Answer with only the number.",
     lambda: cfg("num_key_value_heads")),
    ("vocab",         "In models/hf/SmolLM2-360M-Instruct/config.json, what is vocab_size? "
                      "Answer with only the number.",
     lambda: cfg("vocab_size")),
    ("files_bench",   "How many files are in the bench directory? Answer with only the number.",
     lambda: n_files("bench")),
]


def normalise(s):
    """Accept a number however it is dressed up; accept yes/no in any case."""
    if s is None:
        return None
    s = s.strip().strip(".").strip()
    low = s.lower()
    if low.startswith("yes"): return "yes"
    if low.startswith("no"):  return "no"
    m = re.search(r"-?\d[\d,]*", s)
    return m.group(0).replace(",", "") if m else low[:40]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", default="local")
    ap.add_argument("--label", required=True)
    ap.add_argument("--steps", type=int, default=8)
    a = ap.parse_args()

    print("=" * 80)
    print(f"EVAL · {a.label}")
    print("=" * 80)
    print(f"  {'task':<15}{'expected':>10}{'answered':>12}{'':>4}{'steps':>7}{'sent':>9}{'wall':>8}")
    print("  " + "-" * 68)

    rows, passed, gave_up, wrong = [], 0, 0, 0
    t_all = time.perf_counter()
    for name, question, truth_fn in TASKS:
        truth = normalise(truth_fn())
        r = ag.run_task(a.endpoint, a.model, question, max_steps=a.steps, verbose=False)
        got = normalise(r["answer"])
        ok = (got == truth)
        if not r["ok"]:
            gave_up += 1; mark = "gave up"
        elif ok:
            passed += 1; mark = "PASS"
        else:
            wrong += 1; mark = "wrong"
        rows.append({"task": name, "truth": truth, "got": got, "ok": ok,
                     "answered": r["ok"], "steps": len(r["steps"]),
                     "sent": r["sent"], "out": r["out"], "wall": round(r["wall"], 2)})
        print(f"  {name:<15}{truth:>10}{str(got)[:11]:>12}{'':>1}{mark:<9}"
              f"{len(r['steps']):>4}{r['sent']:>9,}{r['wall']:>7.1f}s")

    n = len(TASKS)
    total_wall = time.perf_counter() - t_all
    print("  " + "-" * 68)
    print()
    print("=" * 80)
    print("SCORE")
    print("=" * 80)
    print(f"  correct            {passed:>3} / {n}   {passed/n*100:>5.1f}%")
    print(f"  answered, wrong    {wrong:>3} / {n}   {wrong/n*100:>5.1f}%   <- indistinguishable from")
    print(f"                                          correct without this harness")
    print(f"  gave up            {gave_up:>3} / {n}   {gave_up/n*100:>5.1f}%")
    print()
    print(f"  completion rate    {(passed+wrong)/n*100:>5.1f}%   <- what a dashboard would show")
    print(f"  ACCURACY           {passed/n*100:>5.1f}%   <- what actually matters")
    print()
    print(f"  tokens sent        {sum(r['sent'] for r in rows):>8,}")
    print(f"  tokens generated   {sum(r['out'] for r in rows):>8,}")
    print(f"  wall clock         {total_wall:>8.1f}s")

    out = {"label": a.label, "model": a.model, "endpoint": a.endpoint,
           "n": n, "correct": passed, "wrong": wrong, "gave_up": gave_up,
           "sent": sum(r["sent"] for r in rows), "out": sum(r["out"] for r in rows),
           "wall": round(total_wall, 1), "rows": rows}
    p = f"{ROOT}/bench/results/eval-{a.label}.json"
    json.dump(out, open(p, "w"), indent=1)
    print(f"\n  -> {p}")


if __name__ == "__main__":
    main()
