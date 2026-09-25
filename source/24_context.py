#!/usr/bin/env python3
"""
24_context.py — how far context actually goes (ch.24).

Chapter 23 bought room: an 8-bit cache holds roughly 15,400 tokens where 16-bit
held 8,192. Room is not the same as usefulness. Two separate questions:

  1. What does a long prompt COST? Prefill is work, and it grows.
  2. Can the model USE it? A context window is a promise about what the model
     may attend to, not a promise that it will.

The second is tested by hiding one fact in a large amount of unrelated text and
asking for it back — at several depths, because where it sits turns out to matter.

    uv run source/24_context.py
"""
import json, os, signal, subprocess, sys, time, urllib.request, urllib.error

ROOT = "/root/Desktop/selfhostllm"
BIN = f"{ROOT}/engines/llamacpp/build/bin/llama-server"
MODEL = f"{ROOT}/models/gguf/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
HOST, PORT = "10.0.0.20", 8089
CTX = 16384                      # what the 8-bit cache from ch.23 makes affordable

FILLER = ("The engine reads every weight for each token it produces. "
          "Memory bandwidth sets the ceiling and the cache decides what is skipped. ")
SECRET = "The maintenance code for the blue generator is 74213."


def start(ctx, ctk="q8_0"):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    p = subprocess.Popen([BIN, "-m", MODEL, "--host", HOST, "--port", str(PORT),
                          "-ngl", "99", "-c", str(ctx), "--no-warmup",
                          "-ctk", ctk, "-ctv", ctk],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         env=env, preexec_fn=os.setsid)
    for _ in range(150):
        time.sleep(1)
        try:
            urllib.request.urlopen(f"http://{HOST}:{PORT}/v1/models", timeout=2)
            return p
        except Exception:
            if p.poll() is not None:
                return None
    return None


def ask(prompt, max_tokens=24, timeout=600):
    body = json.dumps({"model": "l", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0,
                       "stream": True, "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(f"http://{HOST}:{PORT}/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = None; out = []; usage = None
    try:
        r_ = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:160]
        return ("__REFUSED__ " + detail, None, None, None)
    with r_ as r:
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
    return ("".join(out), (first - t0) if first else None,
            time.perf_counter() - t0, (usage or {}).get("prompt_tokens"))


def haystack(n_reps, depth):
    """One fact buried in filler, at a given fraction of the way through."""
    body = FILLER * n_reps
    cut = int(len(body) * depth)
    return (body[:cut] + " " + SECRET + " " + body[cut:] +
            "\n\nWhat is the maintenance code for the blue generator? "
            "Answer with only the number.")


def main():
    print(f"starting engine, context {CTX:,}, 8-bit cache ...", flush=True)
    p = start(CTX)
    if not p:
        sys.exit("engine failed to start")
    try:
        # ---- 1 · what a long prompt costs ------------------------------
        print()
        print("=" * 74)
        print("1 · WHAT A LONG PROMPT COSTS")
        print("=" * 74)
        print(f"  {'prompt tokens':>14}{'time to first':>16}{'per 1k tokens':>16}")
        print("  " + "-" * 46)
        for reps in (10, 100, 300, 600, 1000):
            text = FILLER * reps + "\n\nReply with one word."
            txt, ttft, _, n = ask(text, max_tokens=4)
            if txt.startswith("__REFUSED__"):
                approx = len(text) // 4
                print(f"  ~{approx:>13,}{'REFUSED':>16}   {txt[12:110]}")
                break
            if ttft and n:
                print(f"  {n:>14,}{ttft*1000:>15.0f}ms{ttft/n*1e6:>15.1f}ms")

        # ---- 2 · can it find one fact in all that? ----------------------
        print()
        print("=" * 74)
        print("2 · CAN IT USE WHAT IT WAS GIVEN? — one fact, hidden, then asked for")
        print("=" * 74)
        depths = [0.05, 0.25, 0.5, 0.75, 0.95]
        print(f"  {'context':>9}", end="")
        for d in depths: print(f"{int(d*100):>8}%", end="")
        print(f"{'found':>9}")
        print("  " + "-" * 62)
        grid = []
        for reps in (20, 100, 250, 500, 900):
            row, n_tok = [], None
            for d in depths:
                txt, _, _, n = ask(haystack(reps, d))
                if txt.startswith("__REFUSED__"):
                    row.append(None); continue
                n_tok = n
                row.append("74213" in txt.replace(",", ""))
            if all(h is None for h in row):
                print(f"  {'~'+str(reps*len(FILLER)//4):>9}{'  engine refused this length':<48}")
                grid.append((None, row)); continue
            grid.append((n_tok, row))
            print(f"  {n_tok:>9,}", end="")
            for hit in row:
                print(f"{('OK' if hit else '--') if hit is not None else 'n/a':>9}", end="")
            ok = sum(1 for h in row if h)
            print(f"{ok}/{len(row):<8}")

        print()
        print("=" * 74)
        print("SUMMARY")
        print("=" * 74)
        for n_tok, row in grid:
            if n_tok is None:
                print("  (length refused by the engine)"); continue
            ok = sum(1 for h in row if h)
            print(f"  {n_tok:>7,} tokens: found {ok} of {len(row)} placements")
        json.dump([{"tokens": n, "found": [bool(x) if x is not None else None for x in r]}
                   for n, r in grid],
                  open(f"{ROOT}/bench/results/context-needle.json", "w"), indent=1)
    finally:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL); p.wait(timeout=10)
        except Exception:
            pass


if __name__ == "__main__":
    main()
