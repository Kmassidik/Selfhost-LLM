"""1.00x with speculation running. So the draft is not cheap enough.

A draft only pays if it is MUCH cheaper than the target. Drafting 3 tokens costs
3 draft steps plus 1 target verification; if the draft costs half the target,
that is 2.5 target-steps of work to win at most 3 tokens. The margin vanishes.

This measures how cheap our candidate drafts actually are, and checks whether
the mismatched-vocabulary pairing that llama.cpp ACCEPTED produces correct text.
"""
import json, os, signal, subprocess, time, urllib.request
ROOT = "/root/Desktop/selfhostllm"
BIN = f"{ROOT}/engines/llamacpp/build/bin/llama-server"
M = {"Q8 8B": f"{ROOT}/models/gguf/Meta-Llama-3-8B-Instruct-Q8_0.gguf",
     "Q4 8B": f"{ROOT}/models/gguf/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
     "SmolLM2 360M": f"{ROOT}/models/gguf/SmolLM2-360M-Instruct-F16.gguf"}
HOST, PORT = "10.0.0.20", 8093
PROMPT = ("Write a Python function called read_header that opens a file, reads the "
          "first 8 bytes as a little-endian unsigned 64-bit integer, and returns it. "
          "Include a docstring.")

def launch(args, log):
    p = subprocess.Popen([BIN]+args, stdout=open(log,"w"), stderr=subprocess.STDOUT,
                         preexec_fn=os.setsid)
    for _ in range(180):
        time.sleep(1)
        try: urllib.request.urlopen(f"http://{HOST}:{PORT}/v1/models",timeout=2); return p
        except Exception:
            if p.poll() is not None: return None
    return None

def kill(p):
    if not p: return
    try: os.killpg(os.getpgid(p.pid), signal.SIGKILL); p.wait(timeout=10)
    except Exception: pass
    time.sleep(4)

def gen(n=120):
    body=json.dumps({"model":"l","messages":[{"role":"user","content":PROMPT}],
                     "max_tokens":n,"temperature":0,"stream":True,
                     "stream_options":{"include_usage":True}}).encode()
    req=urllib.request.Request(f"http://{HOST}:{PORT}/v1/chat/completions",body,
                               {"Content-Type":"application/json"})
    t0=time.perf_counter(); first=None; out=[]; usage=None
    with urllib.request.urlopen(req,timeout=900) as r:
        for raw in r:
            l=raw.decode("utf-8","replace").strip()
            if not l.startswith("data:"): continue
            d=l[5:].strip()
            if d=="[DONE]": break
            try: j=json.loads(d)
            except Exception: continue
            if j.get("usage"): usage=j["usage"]
            ch=j.get("choices") or []
            dl=(ch[0].get("delta") or {}).get("content") if ch else None
            if dl:
                if first is None: first=time.perf_counter()
                out.append(dl)
    end=time.perf_counter(); tok=(usage or {}).get("completion_tokens") or len(out)
    return "".join(out), tok/(end-first) if first and end>first else 0

print("A · how cheap is each candidate draft, on its own?")
print(f"  {'model':<16}{'tok/s':>9}{'ngl':>6}")
print("  " + "-"*33)
speeds={}
for name,path in M.items():
    ngl = "28" if "Q8" in name else "99"
    p=launch(["-m",path,"--host",HOST,"--port",str(PORT),"-ngl",ngl,"-c","4096",
              "--no-warmup"], "/tmp/s1.log")
    if not p: print(f"  {name:<16}{'failed':>9}"); continue
    gen(8); txt,tps = gen()
    speeds[name]=tps
    print(f"  {name:<16}{tps:>9.1f}{ngl:>6}")
    kill(p)

if "Q8 8B" in speeds and "Q4 8B" in speeds:
    r = speeds["Q4 8B"]/speeds["Q8 8B"]
    print(f"""
  The Q4 draft is only {r:.1f}x faster than the Q8 target. Drafting 3 tokens
  costs 3 draft steps + 1 verification = {3/r + 1:.1f} target-steps of work, to win
  at most 3 tokens. That is a break-even of about {(3/r+1)/3:.0%} acceptance just to
  not lose — which is why the measurement came back at exactly 1.00x.""")
if "SmolLM2 360M" in speeds and "Q8 8B" in speeds:
    print(f"""
  SmolLM2 is {speeds['SmolLM2 360M']/speeds['Q8 8B']:.0f}x faster and WOULD be a good draft on cost alone.
  It cannot be one: 49,152 tokens against 128,256 — the two models do not agree
  on what any token number means.""")

print()
print("B · llama.cpp ACCEPTED the mismatched draft. Is the output still correct?")
p=launch(["-m",M["Q8 8B"],"--host",HOST,"--port",str(PORT),"-ngl","28","-c","4096",
          "--no-warmup","-md",M["SmolLM2 360M"],"-ngld","99"], "/tmp/s2.log")
if not p:
    print("  refused to start after all")
else:
    gen(8); txt,tps = gen()
    print(f"  ran at {tps:.1f} tok/s")
    print(f"  first 90 chars: {txt.strip()[:90]!r}")
    kill(p)
