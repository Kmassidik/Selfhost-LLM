#!/usr/bin/env python3
"""
30_agent.py — the agentic coding and testing loop this project set out to build.

Chapter 21 built an agent that could read. This one can write a file and run a
test, which is the difference between answering questions about a repository
and changing one.

The loop is the same five steps. What changes is that the outcome is no longer
a claim the model makes about its own work — it is the exit code of a test the
model did not write and cannot edit.

    THE TASK       a source file with a bug, and a test that fails because of it
    THE AGENT      may read, write, list, grep, and run the test
    DONE           the test command exits 0
    FAILED         the budget runs out first

That is the whole design, and the reason it is worth building: chapter 22's
harness scored lookups, and said so — "measures tool use, not reasoning or code
generation". This measures whether a change to a file made a failing test pass.

    uv run source/30_agent.py --endpoint URL --model NAME --label NAME
"""
import argparse, json, os, re, shutil, subprocess, sys, tempfile, time, urllib.request

ROOT = "/root/Desktop/selfhostllm"
TASKS_DIR = f"{ROOT}/agent-tasks"
CMD_TIMEOUT = 20

SYSTEM = """You are fixing a bug in a Python file so that an existing test passes.

Reply with EXACTLY ONE action per message, and nothing else. No explanation.

  LIST
  READ <path>
  WRITE <path>
  ```
  <the complete new contents of the file>
  ```
  TEST
  DONE

Rules:
  - WRITE replaces the whole file. Always READ it first.
  - TEST runs the test suite and shows you the result.
  - Do not edit any file whose name starts with test_.
  - Say DONE only after TEST has passed.
"""


# ── the tools ─────────────────────────────────────────────────────────
# Junk that is not part of the task. These reached the box inside a copy from
# a Mac, and os.walk lists them even though ls does not, so the model was being
# shown ._window.py next to window.py and left to work out which was real.
# Module level, not a class attribute: ignore_patterns returns a plain function
# and Python would hand it self as an extra argument.
JUNK = shutil.ignore_patterns("._*", ".DS_Store", "__pycache__", "*.pyc",
                              ".git", ".pytest_cache")


class Workspace:
    """A copy of the task, so the original is never modified and each attempt
    starts from the same broken state."""

    def __init__(self, task_dir):
        self.dir = tempfile.mkdtemp(prefix="agentwork-")
        shutil.copytree(task_dir, self.dir, dirs_exist_ok=True,
                        ignore=JUNK)

    def _safe(self, rel):
        p = os.path.normpath(os.path.join(self.dir, rel.strip().lstrip("/")))
        return p if p.startswith(self.dir) else None

    def list(self):
        out = []
        for d, ds, fs in os.walk(self.dir):
            ds[:] = [x for x in ds if x not in ("__pycache__", ".git", ".pytest_cache")]
            for f in sorted(fs):
                if f.endswith(".pyc") or f.startswith("._") or f == ".DS_Store":
                    continue
                out.append(os.path.relpath(os.path.join(d, f), self.dir))
        return "\n".join(sorted(out)) or "(empty)"

    def read(self, rel):
        p = self._safe(rel)
        if not p or not os.path.isfile(p):
            return f"error: no such file: {rel.strip()}"
        return open(p, errors="replace").read()[:4000]

    def write(self, rel, body):
        p = self._safe(rel)
        if not p:
            return "refused: path outside the workspace"
        if os.path.basename(p).startswith("test_"):
            return "refused: the tests are not yours to edit"
        before = open(p, errors="replace").read() if os.path.isfile(p) else None
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").write(body)
        if before is not None and before == body:
            # Reporting "wrote 60 bytes" for a write that changed nothing is
            # true and useless. A model that rewrites a file identically and
            # then runs the test reads the same failure as evidence that its
            # edit did not work, and edits again. Saying what actually
            # happened costs nothing and gives away nothing about the bug.
            return (f"wrote {len(body)} bytes to {rel.strip()}, but the "
                    f"contents are identical to what was already there - "
                    f"nothing changed")
        return f"wrote {len(body)} bytes to {rel.strip()}"

    def test(self):
        """The ground truth. Nothing the model says can change this."""
        try:
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "--no-header",
                                "-x", self.dir],
                               capture_output=True, text=True, timeout=CMD_TIMEOUT,
                               cwd=self.dir)
            out = (r.stdout + r.stderr).strip()
            return r.returncode == 0, out[-1200:]
        except subprocess.TimeoutExpired:
            return False, f"the test did not finish within {CMD_TIMEOUT}s"

    def cleanup(self):
        shutil.rmtree(self.dir, ignore_errors=True)


# ── parsing what the model said ───────────────────────────────────────
def parse(reply):
    r = reply.strip()
    # The path must be on the WRITE line itself. "WRITE" alone followed by a
    # newline and the file body used to match \S+ against the first word of
    # that body, so the agent wrote the new contents to a file called "def"
    # and reported success. Llama-3-8B does this often enough to lose tasks on
    # it, and the failure is invisible: the real file is never touched.
    if re.search(r"^[ \t]*WRITE[ \t]*$", r, re.M):
        return ("bad", "WRITE needs the path on the same line, then the "
                       "contents in a fenced block", None)
    m = re.search(r"^[ \t]*WRITE[ \t]+(\S+)[ \t]*$", r, re.M)
    if m:
        body = re.search(r"```(?:python)?\s*\n(.*?)```", r, re.S)
        if not body:
            return ("bad", "WRITE needs the new contents in a fenced block", None)
        return ("write", m.group(1), body.group(1))
    m = re.search(r"^[ \t]*READ[ \t]+(\S+)[ \t]*$", r, re.M)
    if m:
        return ("read", m.group(1), None)
    if re.search(r"^\s*TEST\s*$", r, re.M):
        return ("test", None, None)
    if re.search(r"^\s*LIST\s*$", r, re.M):
        return ("list", None, None)
    if re.search(r"^\s*DONE\s*$", r, re.M):
        return ("done", None, None)
    return ("bad", "no LIST, READ, WRITE, TEST or DONE found", None)


def canonical(kind, a, b, reply):
    """The action alone, with whatever else the model wrote thrown away.

    A model trained to complete code does not stop after "READ window.py" --
    it helpfully writes out what it imagines the file says. Appending that raw
    reply to the conversation puts an invented file into the model's own
    history, one turn before the real contents arrive from the tool, and from
    then on it is arguing with itself about which of the two is the file.
    Qwen3-Coder lost that argument on t1 and rewrote a working last_n() as a
    rectangle class. It had read the real code, twice.

    So the transcript records the action taken and nothing else. A reply that
    parsed to nothing is kept as-is, because there the model needs to see what
    it actually said.
    """
    if kind == "bad":
        return reply[:500]
    act = (f"WRITE {a}\n```python\n{b}```" if kind == "write"
           else f"READ {a}" if kind == "read" else kind.upper())
    # Anything the model wrote BEFORE the action is its reasoning and is kept;
    # only the invented tool output that follows is dropped.
    head = re.split(r"^[ \t]*(?:LIST|READ|WRITE|TEST|DONE)\b", reply, maxsplit=1,
                    flags=re.M)[0].strip()
    return (head + "\n" + act) if head else act


def chat(endpoint, model, messages, timeout=900):
    body = json.dumps({"model": model, "messages": messages,
                       "max_tokens": 700, "temperature": 0}).encode()
    req = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions",
                                 body, {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return (d["choices"][0]["message"]["content"],
            d.get("usage", {}), time.perf_counter() - t0)


def forget(endpoint, slots=8):
    """Erase the engine's cached prefixes before a task starts.

    Without this the benchmark does not measure what it claims to. Every task
    here opens with the same system prompt, so after the first one that prefix
    is in the engine's cache and later tasks are prefilled from it rather than
    computed. Cached and freshly computed prefill do not always produce the
    same logits, and the difference is enough to change the answer: on a cold
    engine t4_twofiles failed six times out of six on its own, and solved in
    five steps every time when t1 to t3 ran first. Same model, same task, same
    code -- only the cache differed.

    So each task starts from the same place, and a task's score means
    something on its own instead of depending on what preceded it.
    """
    root = endpoint.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    for i in range(slots):
        try:
            req = urllib.request.Request(f"{root}/slots/{i}?action=erase",
                                         b"", method="POST")
            urllib.request.urlopen(req, timeout=20).read()
        except Exception:
            # a server started without --slot-save-path answers 501, and a
            # slot index past the end 400. Neither is worth stopping for, but
            # the caller should know the isolation is not in force.
            return False
    return True


def solve(endpoint, model, task_dir, max_steps=14, verbose=True):
    ws = Workspace(task_dir)
    isolated = forget(endpoint)
    brief = open(os.path.join(task_dir, "TASK.md")).read()
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": brief + "\n\nFiles:\n" + ws.list()}]
    sent = out = 0
    wall = 0.0
    tested = False
    try:
        for i in range(max_steps):
            reply, usage, dt = chat(endpoint, model, msgs)
            sent += usage.get("prompt_tokens", 0)
            out += usage.get("completion_tokens", 0)
            wall += dt
            kind, a, b = parse(reply)
            if verbose:
                print(f"    {i+1:>2}. {kind:<6} {(a or '')[:40]:<42} "
                      f"sent {sent:>6,}  {dt:>5.1f}s")
            if kind == "done":
                ok, log = ws.test()
                return {"solved": ok, "steps": i + 1, "sent": sent, "out": out,
                        "wall": wall, "tested": tested, "why": "claimed done",
                        "isolated": isolated}
            if kind == "test":
                tested = True
                ok, log = ws.test()
                if ok:
                    return {"solved": True, "steps": i + 1, "sent": sent,
                            "out": out, "wall": wall, "tested": True,
                            "why": "test passed", "isolated": isolated}
                result = f"The test failed:\n{log}"
            elif kind == "read":
                result = ws.read(a)
            elif kind == "write":
                result = ws.write(a, b)
            elif kind == "list":
                result = ws.list()
            else:
                result = f"That was not a valid action ({a}). Reply with exactly one."
            msgs.append({"role": "assistant",
                         "content": canonical(kind, a, b, reply)})
            msgs.append({"role": "user", "content": result[:2500]})
        return {"solved": False, "steps": max_steps, "sent": sent, "out": out,
                "wall": wall, "tested": tested, "why": "ran out of steps",
                "isolated": isolated}
    finally:
        ws.cleanup()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", default="local")
    ap.add_argument("--label", required=True)
    ap.add_argument("--steps", type=int, default=14)
    a = ap.parse_args()

    tasks = sorted(d for d in os.listdir(TASKS_DIR)
                   if os.path.isdir(os.path.join(TASKS_DIR, d)))
    print("=" * 78)
    print(f"AGENTIC CODING · {a.label} · {len(tasks)} tasks")
    print("=" * 78)
    print("  The model may read, write and run the tests. It may not edit them.")
    print("  A task is solved when the test command exits 0 — nothing else counts.\n")

    rows = []
    for t in tasks:
        print(f"  {t}")
        r = solve(a.endpoint, a.model, os.path.join(TASKS_DIR, t), a.steps)
        r["task"] = t
        rows.append(r)
        print(f"     -> {'SOLVED' if r['solved'] else 'failed'}  ({r['why']})\n")

    n = len(rows)
    solved = sum(1 for r in rows if r["solved"])
    claimed = sum(1 for r in rows if r["why"] == "claimed done" and not r["solved"])
    print("=" * 78)
    print("SCORE")
    print("=" * 78)
    print(f"  {'task':<22}{'solved':>8}{'steps':>7}{'sent':>10}{'wall':>9}  why")
    print("  " + "-" * 74)
    for r in rows:
        print(f"  {r['task']:<22}{('yes' if r['solved'] else 'no'):>8}{r['steps']:>7}"
              f"{r['sent']:>10,}{r['wall']:>8.1f}s  {r['why']}")
    print("  " + "-" * 74)
    print(f"\n  solved            {solved}/{n}   {solved/n*100:.0f}%")
    print(f"  claimed done but the test still failed   {claimed}")
    print(f"  never ran the test at all                {sum(1 for r in rows if not r['tested'])}")
    print(f"\n  total sent {sum(r['sent'] for r in rows):,} tokens, "
          f"generated {sum(r['out'] for r in rows):,}, "
          f"{sum(r['wall'] for r in rows):.0f}s")
    json.dump({"label": a.label, "n": n, "solved": solved, "rows": rows},
              open(f"{ROOT}/bench/results/agent-{a.label}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
