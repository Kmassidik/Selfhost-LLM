#!/usr/bin/env python3
"""
23_kv_quant.py — what quantizing the cache actually costs (ch.23).

Chapter 07 derived the cache formula and noted that dtype bytes is the only
term in it we control. This finds out what happens when we do.

For each cache precision: start the engine, measure what the card holds, score
ten tasks against computed ground truth, and check whether the text changed.

    uv run source/23_kv_quant.py
"""
import json, os, re, signal, subprocess, sys, time, urllib.request, importlib.util

ROOT = "/root/Desktop/selfhostllm"
BIN = f"{ROOT}/engines/llamacpp/build/bin/llama-server"
MODEL = f"{ROOT}/models/gguf/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
HOST, PORT, CTX = "10.0.0.20", 8088, 8192

sys.path.insert(0, f"{ROOT}/source")
_s = importlib.util.spec_from_file_location("ev", f"{ROOT}/source/22_eval.py")
ev = importlib.util.module_from_spec(_s); _s.loader.exec_module(ev)
ag = ev.ag


def vram():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.strip().splitlines()[0])


def start(ctk, ctv):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    p = subprocess.Popen(
        [BIN, "-m", MODEL, "--host", HOST, "--port", str(PORT),
         "-ngl", "99", "-c", str(CTX), "--no-warmup",
         "-ctk", ctk, "-ctv", ctv],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env,
        preexec_fn=os.setsid)
    for _ in range(120):
        time.sleep(1)
        try:
            urllib.request.urlopen(f"http://{HOST}:{PORT}/v1/models", timeout=2)
            return p
        except Exception:
            if p.poll() is not None:
                return None
    return None


def stop(p):
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        p.wait(timeout=10)
    except Exception:
        pass
    time.sleep(3)


def sample_text():
    """A fixed generation, so the text can be compared across cache types."""
    body = json.dumps({"model": "l", "messages": [
        {"role": "user", "content": "Explain in three sentences why reading a "
                                    "model's weights from memory is slower than "
                                    "multiplying them."}],
        "max_tokens": 96, "temperature": 0}).encode()
    r = urllib.request.Request(f"http://{HOST}:{PORT}/v1/chat/completions", body,
                               {"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=300) as resp:
        return json.load(resp)["choices"][0]["message"]["content"]


def main():
    base = vram()
    print(f"card at rest: {base} MB\n")
    print("=" * 84)
    print(f"KV CACHE PRECISION SWEEP · Llama-3-8B Q4_K_M · context {CTX:,} · one card")
    print("=" * 84)
    results = []
    for ctk, ctv in (("f16", "f16"), ("q8_0", "q8_0"), ("q4_0", "q4_0")):
        print(f"\n  --- cache {ctk} ---", flush=True)
        p = start(ctk, ctv)
        if not p:
            print(f"      failed to start"); continue
        time.sleep(2)
        mb = vram()
        text = sample_text()
        ok = 0
        rows = []
        for name, q, truth_fn in ev.TASKS:
            truth = ev.normalise(truth_fn())
            r = ag.run_task(f"http://{HOST}:{PORT}/v1", "l", q, max_steps=8, verbose=False)
            got = ev.normalise(r["answer"])
            good = got == truth
            ok += good
            rows.append({"task": name, "truth": truth, "got": got, "ok": good,
                         "answered": r["ok"]})
        answered = sum(1 for r in rows if r["answered"])
        print(f"      card {mb} MB · correct {ok}/10 · answered {answered}/10", flush=True)
        results.append({"cache": ctk, "mb": mb, "correct": ok,
                        "answered": answered, "text": text, "rows": rows})
        stop(p)

    if not results:
        sys.exit("nothing ran")
    f16 = results[0]
    print()
    print("=" * 84)
    print("WHAT IT COST AND WHAT IT BOUGHT")
    print("=" * 84)
    print(f"  {'cache':<8}{'card MB':>10}{'saved':>9}{'correct':>10}{'answered':>10}"
          f"{'text same as f16':>20}")
    print("  " + "-" * 67)
    for r in results:
        same = "yes" if r["text"] == f16["text"] else "NO"
        print(f"  {r['cache']:<8}{r['mb']:>10,}{f16['mb']-r['mb']:>8,}M"
              f"{r['correct']:>8}/10{r['answered']:>8}/10{same:>20}")
    print()
    # what the formula predicted
    L, kvh, hd = 32, 8, 128
    for r in results:
        bytes_per = {"f16": 2, "q8_0": 1, "q4_0": 0.5}[r["cache"]]
        pred = 2 * L * kvh * hd * bytes_per * CTX / 1e6
        print(f"  cache {r['cache']:<5} predicted by ch.07's formula: {pred:>7.0f} MB")
    json.dump(results, open(f"{ROOT}/bench/results/kv-quant.json", "w"), indent=1)
    print(f"\n  -> {ROOT}/bench/results/kv-quant.json")


if __name__ == "__main__":
    main()
