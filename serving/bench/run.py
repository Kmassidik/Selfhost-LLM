#!/usr/bin/env python3
"""The measurement harness. Every engine in Part I is measured by this file.

One engine, one run, one JSON. The comparison table is generated from those
JSONs and is never hand-written, so it cannot drift from what was measured.

    python3 bench/run.py --endpoint http://10.0.0.20:11434/v1 \
                         --model smollm2:360m --label "L0 ollama bf16"

Six metrics, the same six for every engine:

    ttft_ms            time to first token — what a user feels as lag
    decode_tps         tokens/sec after the first — the streaming speed,
                       from token counts the ENGINE reports, never estimated
    prefill_tps        prompt tokens/sec — a completely different number
    concurrent_tps     total throughput at N parallel requests
    vram_peak_mb       per card, sampled throughout — memory ON the graphics card
    host_rss_peak_mb   resident system RAM of the engine, which nvidia-smi
                       cannot see and which differs hugely between engines
    output_sha256      the completion at temperature 0, hashed

Temperature is 0 everywhere. Without that, two runs cannot be compared at all
and the hash is meaningless.

Only the standard library is used on purpose: the harness must never fail to
run because an engine's environment installed something incompatible.
"""
import argparse, hashlib, json, os, statistics as st, subprocess, sys, threading, time
import urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))


# ── GPU sampling ────────────────────────────────────────────────────
class GpuWatch:
    """Polls memory in the background, BOTH kinds.

    VRAM is what people quote, but an engine's system RAM matters just as much:
    a Python engine with a deep learning framework behind it can hold gigabytes
    of host memory that never appears in nvidia-smi. Sampling only the card
    would report a C++ binary and a Python server as comparable when they are
    not.

    peak      VRAM per card, from nvidia-smi
    peak_rss  resident system memory of the engine process tree, from /proc
    """

    def __init__(self, hz=5, match=None):
        self.interval = 1.0 / hz
        self.peak = {}
        self.peak_rss = 0
        self.match = match or "vllm|sglang|llama-server|ollama"
        self._stop = threading.Event()
        self._t = None

    def _poll(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout
                for line in out.strip().splitlines():
                    i, mb = [x.strip() for x in line.split(",")]
                    self.peak[int(i)] = max(self.peak.get(int(i), 0), int(mb))
            except Exception:
                pass
            try:
                # Sum resident memory across the engine's whole process tree.
                # These servers fork workers, and the parent alone understates it.
                rss = subprocess.run(
                    ["bash", "-c",
                     f"ps -eo rss,comm,args --no-headers | grep -E '{self.match}' "
                     "| grep -v grep | awk '{{s+=$1}} END {{print s+0}}'"],
                    capture_output=True, text=True, timeout=5).stdout.strip()
                self.peak_rss = max(self.peak_rss, int(rss or 0) // 1024)
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self._t = threading.Thread(target=self._poll, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        if self._t:
            self._t.join(timeout=2)


# ── one request, streamed so the first token can be timed ───────────
def stream_once(endpoint, model, prompt, max_tokens, temperature, timeout=600):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        # Ask the engine for REAL token counts. Estimating from character
        # length was wrong by 1.43x on the first engine tested, which would
        # have put every tok/s figure in this project out by nearly half.
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=body, headers={"Content-Type": "application/json"})

    t0 = time.perf_counter()
    first = None
    pieces = []
    usage = None
    chunks = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                d = json.loads(data)
            except json.JSONDecodeError:
                continue
            if d.get("usage"):
                usage = d["usage"]
            ch = d.get("choices") or []
            delta = (ch[0].get("delta") or {}).get("content") if ch else None
            if delta:
                if first is None:
                    first = time.perf_counter()
                pieces.append(delta)
                chunks += 1
    end = time.perf_counter()
    text = "".join(pieces)

    # Token count, best source first. Which source was used is recorded, because
    # a number is only as trustworthy as where it came from.
    if usage and usage.get("completion_tokens"):
        out_tok, source = usage["completion_tokens"], "engine"
    elif chunks:
        out_tok, source = chunks, "chunks"      # one SSE chunk per token, typically
    else:
        out_tok, source = len(text) / 4, "chars/4"   # last resort, and it is poor
    return {
        "ttft_s": (first - t0) if first else None,
        "total_s": end - t0,
        "decode_s": (end - first) if first else None,
        "text": text,
        "chars": len(text),
        "out_tokens": out_tok,
        "in_tokens": (usage or {}).get("prompt_tokens"),
        "token_source": source,
    }


def expand(text):
    """REPEAT:n:phrase -> phrase repeated n times, then the rest."""
    if not text.startswith("REPEAT:"):
        return text
    _, n, rest = text.split(":", 2)
    filler, tail = rest.split("\n\n", 1)
    return filler * int(n) + "\n\n" + tail


# ── the run ─────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", required=True, help="OpenAI-compatible base URL, e.g. .../v1")
    p.add_argument("--model", required=True)
    p.add_argument("--label", required=True, help="what is being measured, e.g. 'L0 ollama bf16'")
    p.add_argument("--level", default="", help="L0 / L1 / L2 ...")
    p.add_argument("--quant", default="bf16")
    p.add_argument("--repeats", type=int, default=3, help="per prompt; the median is kept")
    p.add_argument("--concurrency", default="1,4,16", help="batch sizes for the throughput test")
    p.add_argument("--skip-concurrent", action="store_true")
    p.add_argument("--note", default="")
    a = p.parse_args()

    cfg = json.load(open(os.path.join(HERE, "prompts.json")))
    run_id = f"{a.level or 'x'}-{a.model.replace('/','_').replace(':','_')}-{int(time.time())}"
    print(f"{a.label}\n  endpoint {a.endpoint}\n  model    {a.model}\n  run id   {run_id}\n")

    result = {
        "run_id": run_id, "label": a.label, "level": a.level, "model": a.model,
        "quant": a.quant, "endpoint": a.endpoint, "note": a.note,
        "prompt_set_version": cfg["version"], "temperature": cfg["temperature"],
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "prompts": {}, "concurrency": {},
    }

    with GpuWatch() as gpu:
        # per-prompt, single stream
        for spec in cfg["prompts"]:
            text = expand(spec["text"])
            runs = []
            for i in range(a.repeats):
                try:
                    r = stream_once(a.endpoint, a.model, text,
                                    spec["max_tokens"], cfg["temperature"])
                except Exception as e:
                    print(f"  {spec['id']:<9} FAILED: {type(e).__name__}: {e}")
                    result["prompts"][spec["id"]] = {"error": f"{type(e).__name__}: {e}"}
                    runs = []
                    break
                runs.append(r)
            if not runs:
                continue

            ttfts = [r["ttft_s"] for r in runs if r["ttft_s"]]
            out_tok = st.median([r["out_tokens"] for r in runs])
            decode = st.median([r["decode_s"] for r in runs if r["decode_s"]])
            in_tok = runs[0].get("in_tokens")
            prefill_s = st.median(ttfts) if ttfts else None
            entry = {
                "ttft_ms": round(st.median(ttfts) * 1000, 2) if ttfts else None,
                "total_s": round(st.median([r["total_s"] for r in runs]), 3),
                "decode_s": round(decode, 3),
                "out_tokens": out_tok,
                "in_tokens": in_tok,
                "token_source": runs[0]["token_source"],
                "decode_tps": round(out_tok / decode, 1) if decode else None,
                "prefill_tps": round(in_tok / prefill_s, 1) if (in_tok and prefill_s) else None,
                "prompt_chars": len(text),
                "output_sha256": hashlib.sha256(runs[0]["text"].encode()).hexdigest()[:16],
                "output_head": runs[0]["text"][:120],
                "repeats": len(runs),
            }
            result["prompts"][spec["id"]] = entry
            print(f"  {spec['id']:<9} ttft {entry['ttft_ms'] or 0:>7.1f} ms   "
                  f"{entry['decode_tps'] or 0:>6.1f} tok/s   "
                  f"{entry['out_tokens']:>4.0f} tok ({entry['token_source']})   "
                  f"hash {entry['output_sha256']}")

        # concurrency
        if not a.skip_concurrent:
            spec = next(s for s in cfg["prompts"] if s["id"] == "medium")
            print()
            for n in [int(x) for x in a.concurrency.split(",")]:
                t0 = time.perf_counter()
                try:
                    with ThreadPoolExecutor(max_workers=n) as ex:
                        rs = list(ex.map(lambda _: stream_once(
                            a.endpoint, a.model, spec["text"],
                            spec["max_tokens"], cfg["temperature"]), range(n)))
                except Exception as e:
                    print(f"  concurrency {n:<4} FAILED: {type(e).__name__}")
                    break
                wall = time.perf_counter() - t0
                tok = sum(r["out_tokens"] for r in rs)
                result["concurrency"][str(n)] = {
                    "wall_s": round(wall, 3),
                    "total_tps": round(tok / wall, 1),
                    "per_stream_tps": round(tok / wall / n, 1),
                    "tokens": tok,
                }
                print(f"  concurrency {n:<4} {tok/wall:>7.1f} tok/s total   "
                      f"{tok/wall/n:>6.1f} per stream   ({tok:.0f} tokens in {wall:.2f}s)")

    result["vram_peak_mb"] = gpu.peak
    result["host_rss_peak_mb"] = gpu.peak_rss
    result["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    out = os.path.join(HERE, "results", run_id + ".json")
    json.dump(result, open(out, "w"), indent=2)
    print(f"\n  peak VRAM per card : {gpu.peak}  (on the graphics card)")
    print(f"  peak system RAM    : {gpu.peak_rss:,} MB  (host memory, engine process tree)")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
