#!/usr/bin/env python3
"""
20_agent_traffic.py — what an agent does to a server (ch.20).

Part I measured single questions. An agent is not a single question: it is a
conversation that grows, where every turn re-sends everything said so far
because the server remembers nothing between requests.

This measures the shape of that growth, and what it costs.

    uv run source/20_agent_traffic.py --endpoint http://10.0.0.20:11434/v1
"""
import argparse, json, time, urllib.request, sys

def post(endpoint, body, timeout=300):
    req = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions",
                                 json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return d, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", default="smollm2:360m")
    ap.add_argument("--turns", type=int, default=8)
    a = ap.parse_args()

    print("=" * 76)
    print("1 · A CONVERSATION, TURN BY TURN — what the server receives each time")
    print("=" * 76)
    print(f"  {'turn':>5}{'sent (tokens)':>16}{'new this turn':>16}{'time':>10}{'cumulative sent':>18}")
    print("  " + "-" * 68)

    messages = []
    asks = [
        "List the files you would check first in an unfamiliar Python repository.",
        "Why that order?", "What would you look for in the first one?",
        "And if it is missing?", "Give one concrete example.",
        "How would you verify it?", "What could go wrong?", "Summarise in one line.",
    ]
    cumulative, prev_sent = 0, 0
    rows = []
    for t in range(a.turns):
        messages.append({"role": "user", "content": asks[t % len(asks)]})
        d, dt = post(a.endpoint, {"model": a.model, "messages": messages,
                                  "max_tokens": 60, "temperature": 0})
        u = d.get("usage") or {}
        sent = u.get("prompt_tokens", 0)
        cumulative += sent
        rows.append((t + 1, sent, dt))
        print(f"  {t+1:>5}{sent:>16,}{sent - prev_sent:>16,}{dt:>9.2f}s{cumulative:>18,}")
        prev_sent = sent
        messages.append({"role": "assistant",
                         "content": d["choices"][0]["message"]["content"]})

    first, last = rows[0][1], rows[-1][1]
    print()
    print(f"  Turn 1 sent {first:,} tokens. Turn {a.turns} sent {last:,} — {last/first:.1f}x as many,")
    print(f"  for a question of the same size. The server is stateless, so the whole")
    print(f"  conversation is re-sent every time.")
    print(f"  Total across {a.turns} turns: {cumulative:,} tokens sent to say {a.turns} things.")

    # ---- 2 · what the growth actually is ----------------------------------
    print()
    print("=" * 76)
    print("2 · THE SHAPE OF IT")
    print("=" * 76)
    ideal = sum(r[1] - rows[max(0, i-1)][1] for i, r in enumerate(rows))
    print(f"  tokens a stateful server would need   {ideal:>10,}   (only what is new)")
    print(f"  tokens this stateless one received    {cumulative:>10,}")
    print(f"  waste factor                          {cumulative/max(ideal,1):>10.1f}x")
    print()
    print("  Every turn re-reads the whole conversation. Turn n costs about n times")
    print("  turn one, so the total for a conversation of n turns grows with n squared")
    print("  rather than with n. This is why an agent is a pathological workload and")
    print("  why chapter 13's prefix caching exists.")

    print()
    print("=" * 76)
    print("3 · WHERE THE TIME WENT")
    print("=" * 76)
    print(f"  {'turn':>5}{'sent':>10}{'time':>10}{'ms per 1k sent':>18}")
    print("  " + "-" * 45)
    for t, sent, dt in rows:
        print(f"  {t:>5}{sent:>10,}{dt:>9.2f}s{dt/max(sent,1)*1000*1000:>17.1f}")


if __name__ == "__main__":
    main()
