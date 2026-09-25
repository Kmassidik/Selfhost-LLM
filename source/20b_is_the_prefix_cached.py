#!/usr/bin/env python3
"""
20b_is_the_prefix_cached.py — the server remembered something (ch.20).

A conversation re-sends everything every turn, so the time per turn should grow
with the conversation. In the measurement above it did not: flat at 0.24s while
the prompt tripled.

Either the prompt is too small to matter, or the server is not recomputing the
part it has seen before. Those predict different things, so they can be told
apart: send prompts of the SAME growing length, but change whether the start of
them is repeated or fresh.
"""
import json, time, urllib.request, sys

EP = "http://10.0.0.20:11434/v1/chat/completions"
MODEL = "smollm2:360m"

def ask(messages, timeout=300):
    body = {"model": MODEL, "messages": messages, "max_tokens": 8, "temperature": 0}
    req = urllib.request.Request(EP, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return d.get("usage", {}).get("prompt_tokens", 0), time.perf_counter() - t0

FILLER = "The engine reads every weight for each token it produces. "

# warm the model so loading does not pollute the first row
ask([{"role": "user", "content": "hi"}])

print("Same prompt lengths. The only difference is whether the beginning repeats.")
print()
print(f"  {'prompt tokens':>14}{'shared prefix':>16}{'fresh prefix':>16}{'ratio':>9}")
print("  " + "-" * 57)
shared_base = FILLER * 1     # grows by appending — the start never changes
for n in (1, 20, 60, 120, 240):
    shared = FILLER * n + "\n\nReply with one word."
    s_tok, s_dt = ask([{"role": "user", "content": shared}])
    ask([{"role": "user", "content": shared}])            # prime it
    s_tok, s_dt = ask([{"role": "user", "content": shared}])   # now measure

    fresh = f"[{time.time_ns()}] " + FILLER * n + "\n\nReply with one word."
    f_tok, f_dt = ask([{"role": "user", "content": fresh}])
    print(f"  {s_tok:>14,}{s_dt:>15.3f}s{f_dt:>15.3f}s{f_dt/max(s_dt,1e-9):>8.2f}x")

print()
print("""  A shared prefix that has already been seen costs far less than the same
  number of tokens the server has not seen before. The server is keeping the
  keys and values it computed last time and reusing them — prefix caching,
  which is what chapter 13 credited SGLang for and which Ollama is doing here
  without being asked.

  That is why the conversation above stayed flat: turn n repeats turn n-1's
  prompt exactly, so almost none of it is recomputed. The tokens are still
  SENT, and the waste factor is real, but they are not all re-read.""")
