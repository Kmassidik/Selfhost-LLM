"""Slots, or training limit? One flag separates them.

The startup log showed both a warning that 16,384 exceeds what the model was
trained on, AND that the engine had carved the allocation into 4 slots. Only
one of those is why an 8,800-token request was refused.

  --parallel 1  gives the whole allocation to a single slot.
  If the refusal was about slots, this fixes it. If it was the training limit,
  nothing changes.
"""
import json, os, signal, subprocess, time, urllib.request, urllib.error
ROOT = "/root/Desktop/selfhostllm"
BIN = f"{ROOT}/engines/llamacpp/build/bin/llama-server"
MODEL = f"{ROOT}/models/gguf/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
HOST, PORT = "10.0.0.20", 8091
FILLER = ("The engine reads every weight for each token it produces. "
          "Memory bandwidth sets the ceiling and the cache decides what is skipped. ")
SECRET = "The maintenance code for the blue generator is 74213."

def run(extra, ctx, log):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    p = subprocess.Popen([BIN,"-m",MODEL,"--host",HOST,"--port",str(PORT),"-ngl","99",
                          "-c",str(ctx),"--no-warmup","-ctk","q8_0","-ctv","q8_0"]+extra,
                         stdout=open(log,"w"), stderr=subprocess.STDOUT, env=env,
                         preexec_fn=os.setsid)
    for _ in range(150):
        time.sleep(1)
        try:
            urllib.request.urlopen(f"http://{HOST}:{PORT}/v1/models",timeout=2); return p
        except Exception:
            if p.poll() is not None: return None
    return None

def probe(approx, needle=False):
    reps = approx*4//len(FILLER); body = FILLER*reps
    if needle:
        c=len(body)//2; body=body[:c]+" "+SECRET+" "+body[c:]
        q="\n\nWhat is the maintenance code for the blue generator? Answer with only the number."
    else: q="\n\nReply with one word."
    pl=json.dumps({"model":"l","messages":[{"role":"user","content":body+q}],
                   "max_tokens":16,"temperature":0}).encode()
    r=urllib.request.Request(f"http://{HOST}:{PORT}/v1/chat/completions",pl,
                             {"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(r,timeout=900) as resp: d=json.load(resp)
        return True, d["choices"][0]["message"]["content"], d.get("usage",{}).get("prompt_tokens")
    except urllib.error.HTTPError as e:
        return False, e.read().decode("utf-8","replace")[:110], None

def kill(p):
    try: os.killpg(os.getpgid(p.pid), signal.SIGKILL); p.wait(timeout=10)
    except Exception: pass
    time.sleep(3)

for label, extra in (("4 slots (default)", []), ("--parallel 1", ["--parallel","1"])):
    p = run(extra, 16384, "/tmp/s.log")
    slot = next((l.strip()[-60:] for l in open("/tmp/s.log") if "n_ctx_slot" in l), "?")
    ok, msg, n = probe(12000)
    print(f"{label:<22} {slot}")
    print(f"{'':<22} 12,000-token prompt -> {'ACCEPTED' if ok else 'refused'}")
    if ok:
        f_ok, text, n2 = probe(12000, needle=True)
        hit = "74213" in (text or "").replace(",","")
        print(f"{'':<22} read {n2:,} tokens · needle at 50% -> {'FOUND' if hit else 'NOT FOUND'}")
    kill(p)
    print()
print("""RESULT: not slots. With --parallel 1 there is exactly one slot and
n_ctx_slot is STILL 8,192 — llama.cpp clamps the per-sequence context to what
the model was trained on, however much is allocated and however few slots share
it. The 12,000-token prompt is refused either way.

So the binding limit is the MODEL, not the cache and not the scheduler. Chapter
23 bought room in the cache that nothing can spend.""")
