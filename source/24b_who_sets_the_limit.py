"""The cache had room and the engine refused anyway. Who is imposing the limit?

Chapter 23 concluded an 8-bit cache buys "a longer conversation". The engine
disagreed: asked for 16,384 it reported 8,192 available. Two candidates —
llama.cpp capping at what the model was TRAINED on, or the cache running out.
The second is already ruled out by chapter 23's numbers, so this checks the first
and then tries to get past it.
"""
import json, os, signal, subprocess, sys, time, urllib.request, urllib.error

ROOT = "/root/Desktop/selfhostllm"
BIN = f"{ROOT}/engines/llamacpp/build/bin/llama-server"
MODEL = f"{ROOT}/models/gguf/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
HOST, PORT = "10.0.0.20", 8090
FILLER = ("The engine reads every weight for each token it produces. "
          "Memory bandwidth sets the ceiling and the cache decides what is skipped. ")
SECRET = "The maintenance code for the blue generator is 74213."


def run(extra, ctx, log):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    p = subprocess.Popen([BIN, "-m", MODEL, "--host", HOST, "--port", str(PORT),
                          "-ngl", "99", "-c", str(ctx), "--no-warmup",
                          "-ctk", "q8_0", "-ctv", "q8_0"] + extra,
                         stdout=open(log, "w"), stderr=subprocess.STDOUT,
                         env=env, preexec_fn=os.setsid)
    for _ in range(150):
        time.sleep(1)
        try:
            urllib.request.urlopen(f"http://{HOST}:{PORT}/v1/models", timeout=2); return p
        except Exception:
            if p.poll() is not None: return None
    return None


def probe(n_tokens_approx, want_needle=False):
    reps = n_tokens_approx * 4 // len(FILLER)
    body = FILLER * reps
    if want_needle:
        cut = len(body) // 2
        body = body[:cut] + " " + SECRET + " " + body[cut:]
        q = "\n\nWhat is the maintenance code for the blue generator? Answer with only the number."
    else:
        q = "\n\nReply with one word."
    payload = json.dumps({"model": "l", "messages": [{"role": "user", "content": body + q}],
                          "max_tokens": 16, "temperature": 0}).encode()
    req = urllib.request.Request(f"http://{HOST}:{PORT}/v1/chat/completions", payload,
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            d = json.load(r)
        return True, d["choices"][0]["message"]["content"], d.get("usage", {}).get("prompt_tokens")
    except urllib.error.HTTPError as e:
        return False, e.read().decode("utf-8", "replace")[:120], None


def kill(p):
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL); p.wait(timeout=10)
    except Exception: pass
    time.sleep(3)


print("A · what does the engine say it will allow, asked for 16,384?")
p = run([], 16384, "/tmp/a.log")
ok, msg, n = probe(12000)
print(f"    12,000-token prompt -> {'accepted' if ok else 'REFUSED'}")
if not ok: print(f"    {msg}")
kill(p)
print("    from the server's own startup log:")
for line in open("/tmp/a.log"):
    if "n_ctx" in line and ("train" in line or "per_seq" in line or "slot" in line):
        print(f"      {line.strip()[:110]}")

print()
print("B · the same, with rope scaling switched on to extend beyond training")
p = run(["--rope-scaling", "linear", "--rope-scale", "2"], 16384, "/tmp/b.log")
if not p:
    print("    engine failed to start")
else:
    ok, msg, n = probe(12000)
    print(f"    12,000-token prompt -> {'accepted' if ok else 'REFUSED'}")
    if ok:
        found, text, n2 = probe(12000, want_needle=True)
        hit = "74213" in (text or "").replace(",", "")
        print(f"    prompt tokens actually read: {n2:,}")
        print(f"    needle at 50% depth -> {'FOUND' if hit else 'NOT FOUND'}  {text.strip()[:60]!r}")
    else:
        print(f"    {msg}")
    kill(p)
print("""
A context window is two limits stacked: what the cache can hold, and what the
model was trained to make sense of. Chapter 23 moved the first one. Only the
second one was ever binding.""")
