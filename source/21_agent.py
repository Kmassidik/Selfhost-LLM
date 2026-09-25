#!/usr/bin/env python3
"""
21_agent.py — an agent loop, written by hand (ch.21).

An agent is a loop, and the loop is short enough to fit on a page:

    1. tell the model what tools exist
    2. send the conversation
    3. read the reply. if it asks for a tool, run it
    4. append the result to the conversation and go to 2
    5. stop when it answers instead of asking

Everything difficult is in step 3. The model produces text, and text has to be
turned into an action — which means parsing something a language model wrote,
which means it will eventually be wrong in a way nothing anticipated.

Written by hand rather than pulled from a framework so those failures are
visible instead of being swallowed somewhere.

    uv run source/21_agent.py --endpoint http://10.0.0.20:8086/v1 --model llama3
"""
import argparse, json, os, re, subprocess, time, urllib.request

ROOT = "/root/Desktop/selfhostllm"

TOOLS = """You have tools. To use one, reply with ONLY a line of this form and nothing else:

TOOL: list_files DIR
TOOL: read_file PATH
TOOL: grep PATTERN PATH
TOOL: count_lines PATH

When you have the answer, reply with ONLY:

ANSWER: <your answer>

Rules: one tool per reply. Never explain. Never use a tool and give an answer
in the same reply. Paths are relative to the repository root."""


def tool_list_files(d):
    p = os.path.normpath(os.path.join(ROOT, d.strip()))
    if not p.startswith(ROOT):
        return "refused: path outside the repository"
    try:
        return "\n".join(sorted(os.listdir(p))[:60]) or "(empty)"
    except Exception as e:
        return f"error: {e}"


def tool_read_file(p):
    f = os.path.normpath(os.path.join(ROOT, p.strip()))
    if not f.startswith(ROOT):
        return "refused: path outside the repository"
    try:
        return open(f).read()[:2000]
    except Exception as e:
        return f"error: {e}"


def tool_grep(rest):
    parts = rest.strip().split(None, 1)
    if len(parts) < 2:
        return "error: grep needs a pattern and a path"
    pat, p = parts
    f = os.path.normpath(os.path.join(ROOT, p.strip()))
    if not f.startswith(ROOT):
        return "refused: path outside the repository"
    try:
        out = subprocess.run(["grep", "-rn", pat, f], capture_output=True,
                             text=True, timeout=10).stdout
        return out[:1500] or "(no matches)"
    except Exception as e:
        return f"error: {e}"


def tool_count_lines(p):
    f = os.path.normpath(os.path.join(ROOT, p.strip()))
    if not f.startswith(ROOT):
        return "refused: path outside the repository"
    try:
        return str(sum(1 for _ in open(f)))
    except Exception as e:
        return f"error: {e}"


HANDLERS = {"list_files": tool_list_files, "read_file": tool_read_file,
            "grep": tool_grep, "count_lines": tool_count_lines}


def parse(reply):
    """Turn what the model said into an action. Deliberately forgiving."""
    m = re.search(r"ANSWER:\s*(.+)", reply, re.S)
    if m and "TOOL:" not in reply[:m.start()]:
        return ("answer", m.group(1).strip()[:300], None)
    m = re.search(r"TOOL:\s*(\w+)\s*(.*)", reply)
    if m:
        name = m.group(1)
        if name not in HANDLERS:
            return ("bad", f"no such tool: {name}", None)
        return ("tool", name, m.group(2).split("\n")[0])
    return ("bad", "no TOOL: or ANSWER: line found", None)


def chat(endpoint, model, messages, timeout=600):
    body = {"model": model, "messages": messages, "max_tokens": 160,
            "temperature": 0}
    req = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions",
                                 json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return (d["choices"][0]["message"]["content"],
            d.get("usage", {}), time.perf_counter() - t0)


def run_task(endpoint, model, task, max_steps=8, verbose=True):
    messages = [{"role": "system", "content": TOOLS},
                {"role": "user", "content": task}]
    steps, sent_total, out_total, wall = [], 0, 0, 0.0
    for i in range(max_steps):
        reply, usage, dt = chat(endpoint, model, messages)
        sent = usage.get("prompt_tokens", 0)
        out = usage.get("completion_tokens", 0)
        sent_total += sent; out_total += out; wall += dt
        kind, a, b = parse(reply)
        steps.append({"step": i + 1, "sent": sent, "out": out, "s": round(dt, 2),
                      "kind": kind, "what": a if kind != "answer" else "ANSWER"})
        if verbose:
            label = a if kind != "tool" else f"{a} {b or ''}".strip()
            print(f"    {i+1:>2}. sent {sent:>6,}  out {out:>4}  {dt:>6.2f}s  "
                  f"{kind:<7} {label[:52]}")
        if kind == "answer":
            return {"ok": True, "answer": a, "steps": steps,
                    "sent": sent_total, "out": out_total, "wall": wall}
        if kind == "bad":
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user",
                             "content": f"That was not a valid reply ({a}). "
                                        "Reply with exactly one TOOL: or ANSWER: line."})
            continue
        result = HANDLERS[a](b or "")
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": f"Result:\n{result[:1200]}"})
    return {"ok": False, "answer": None, "steps": steps,
            "sent": sent_total, "out": out_total, "wall": wall}


TASKS = [
    ("count_source", "How many files are in the source directory? "
                     "Answer with only the number."),
    ("find_bench",   "Does the bench directory contain a file called run.py? "
                     "Answer yes or no."),
    ("read_prompts", "How many prompts are listed in bench/prompts.json? "
                     "Answer with only the number."),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", default="local")
    ap.add_argument("--label", default="model")
    a = ap.parse_args()

    print("=" * 78)
    print(f"AGENT LOOP · {a.label} · {a.endpoint}")
    print("=" * 78)
    results = []
    for name, task in TASKS:
        print(f"\n  task: {task}")
        r = run_task(a.endpoint, a.model, task)
        r["name"] = name
        results.append(r)
        print(f"    -> {'ANSWERED' if r['ok'] else 'GAVE UP'}: "
              f"{(r['answer'] or '')[:60]!r}")

    print()
    print("=" * 78)
    print("ACCOUNTING")
    print("=" * 78)
    print(f"  {'task':<15}{'steps':>7}{'sent':>10}{'generated':>11}{'wall':>9}  outcome")
    print("  " + "-" * 64)
    for r in results:
        print(f"  {r['name']:<15}{len(r['steps']):>7}{r['sent']:>10,}{r['out']:>11,}"
              f"{r['wall']:>8.1f}s  {'answered' if r['ok'] else 'gave up'}")
    ts = sum(r["sent"] for r in results); to = sum(r["out"] for r in results)
    tw = sum(r["wall"] for r in results)
    print("  " + "-" * 64)
    print(f"  {'TOTAL':<15}{sum(len(r['steps']) for r in results):>7}{ts:>10,}{to:>11,}{tw:>8.1f}s")
    print()
    print(f"  {to:,} tokens generated, {ts:,} sent — the model was READ TO "
          f"{ts/max(to,1):.0f}x more than it spoke.")
    bad = sum(1 for r in results for s in r["steps"] if s["kind"] == "bad")
    print(f"  replies that could not be parsed as an action: {bad} of "
          f"{sum(len(r['steps']) for r in results)}")
    json.dump(results, open(f"/tmp/agent-{a.label}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
