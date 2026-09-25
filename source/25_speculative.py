#!/usr/bin/env python3
"""
25_speculative.py — two models, one answer (ch.25).

Decoding is serial: each token needs the one before it, so a model produces
them one at a time however much hardware is waiting. Chapter 08 measured the
consequence — a single stream never gets faster no matter how much is batched
alongside it.

Speculative decoding is the trick that breaks that. A cheap DRAFT model guesses
several tokens ahead. The expensive TARGET model then checks all the guesses in
ONE pass, because verifying tokens it already has is a prefill, and prefill is
parallel. Every guess that survives is a token the target got for free.

Two things have to be true, and the first one eliminates most pairings.

    uv run source/25_speculative.py
"""
import json, os, re, signal, subprocess, sys, time, urllib.request, urllib.error

ROOT = "/root/Desktop/selfhostllm"
BIN = f"{ROOT}/engines/llamacpp/build/bin/llama-server"
Q8 = f"{ROOT}/models/gguf/Meta-Llama-3-8B-Instruct-Q8_0.gguf"
Q4 = f"{ROOT}/models/gguf/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
SMOL = f"{ROOT}/models/gguf/SmolLM2-360M-Instruct-F16.gguf"
HOST, PORT = "10.0.0.20", 8092

PROMPT = ("Write a Python function called read_header that opens a file, reads the "
          "first 8 bytes as a little-endian unsigned 64-bit integer, and returns it. "
          "Include a docstring.")


def launch(args, log, wait=180):
    p = subprocess.Popen([BIN] + args, stdout=open(log, "w"),
                         stderr=subprocess.STDOUT, preexec_fn=os.setsid,
                         env=dict(os.environ))
    for _ in range(wait):
        time.sleep(1)
        try:
            urllib.request.urlopen(f"http://{HOST}:{PORT}/v1/models", timeout=2)
            return p
        except Exception:
            if p.poll() is not None:
                return None
    return None


def kill(p):
    if not p: return
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL); p.wait(timeout=10)
    except Exception: pass
    time.sleep(4)


def generate(n=160):
    body = json.dumps({"model": "l", "messages": [{"role": "user", "content": PROMPT}],
                       "max_tokens": n, "temperature": 0, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(f"http://{HOST}:{PORT}/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = None; out = []; usage = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"): continue
            d = line[5:].strip()
            if d == "[DONE]": break
            try: j = json.loads(d)
            except Exception: continue
            if j.get("usage"): usage = j["usage"]
            ch = j.get("choices") or []
            delta = (ch[0].get("delta") or {}).get("content") if ch else None
            if delta:
                if first is None: first = time.perf_counter()
                out.append(delta)
    end = time.perf_counter()
    tok = (usage or {}).get("completion_tokens") or len(out)
    return {"text": "".join(out), "tokens": tok,
            "decode_s": end - first if first else None,
            "tps": tok / (end - first) if first and end > first else 0}


BASE = ["-m", Q8, "--host", HOST, "--port", str(PORT), "-ngl", "28",
        "-c", "4096", "--no-warmup", "-ctk", "q8_0", "-ctv", "q8_0"]

print("=" * 78)
print("1 · THE CONSTRAINT NOBODY MENTIONS — the draft must share the vocabulary")
print("=" * 78)
print("  Trying SmolLM2-360M (vocabulary 49,152) as the draft for")
print("  Llama-3-8B (vocabulary 128,256) ...", flush=True)
p = launch(BASE + ["-md", SMOL, "-ngld", "99", "-devd", "CUDA1"], "/tmp/spec_bad.log", wait=90)
if p:
    print("  it started — unexpected"); kill(p)
else:
    txt = open("/tmp/spec_bad.log").read()
    m = [l.strip() for l in txt.splitlines()
         if re.search(r"vocab|draft|incompat|differ", l, re.I)][-3:]
    print("  REFUSED. The engine said:")
    for l in m: print(f"    {l[:110]}")
print("""
  A draft is only useful if its guesses can be CHECKED against the target's
  own probabilities, and that requires both models to mean the same thing by
  token number 4,721. Different tokenizer, no pairing — which rules out almost
  every small model as a draft for almost every large one.
""")

print("=" * 78)
print("2 · A PAIR THAT DOES MATCH — same model, different precision")
print("=" * 78)
results = {}
for label, extra, log in (
        ("target alone (Q8)", [], "/tmp/spec_a.log"),
        ("Q8 target + Q4 draft", ["-md", Q4, "-ngld", "99", "-devd", "CUDA1"], "/tmp/spec_b.log")):
    print(f"\n  {label} ...", flush=True)
    p = launch(BASE + extra, log)
    if not p:
        print(f"    failed to start — see {log}"); continue
    generate(16)                       # warm
    r = generate()
    results[label] = r
    print(f"    {r['tokens']} tokens in {r['decode_s']:.2f}s = {r['tps']:.1f} tok/s")
    # llama.cpp reports draft acceptance in its log
    acc = [l.strip() for l in open(log) if re.search(r"draft acceptance|n_drafted|n_accept", l, re.I)]
    for l in acc[-3:]:
        print(f"    {l[-100:]}")
    kill(p)

if len(results) == 2:
    a = results["target alone (Q8)"]; b = results["Q8 target + Q4 draft"]
    print()
    print("=" * 78)
    print("3 · WHAT IT BOUGHT")
    print("=" * 78)
    print(f"  {'configuration':<26}{'tok/s':>10}{'speedup':>10}{'output identical':>20}")
    print("  " + "-" * 66)
    print(f"  {'Q8 alone':<26}{a['tps']:>10.1f}{1.0:>10.2f}x{'baseline':>20}")
    same = "yes" if a["text"] == b["text"] else "NO"
    print(f"  {'Q8 with a Q4 draft':<26}{b['tps']:>10.1f}{b['tps']/a['tps']:>9.2f}x{same:>20}")
    print()
    print("  Speculative decoding is only worth anything if the output is unchanged.")
    print("  The target VERIFIES every guess, so a wrong guess is discarded rather")
    print("  than accepted — the result must match what the target would have said")
    print("  alone, or the implementation is broken rather than fast.")
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "text"}
               for k, v in results.items()},
              open(f"{ROOT}/bench/results/speculative.json", "w"), indent=1)
