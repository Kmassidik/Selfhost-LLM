#!/usr/bin/env python3
"""Rebuild the comparison table from bench/results/*.json.

The table is never hand-maintained. It is generated from the measurements
every time, so it cannot drift from what was actually recorded — and a run
that was not measured cannot appear in it.

    python3 bench/table.py            the comparison
    python3 bench/table.py --md       markdown, for pasting into a chapter
    python3 bench/table.py --full     every metric of every run
"""
import glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
runs = []
# results/ also holds analysis files written by chapter scripts — kv-quant.json,
# corpus.json, the-run.json — which are not benchmark runs. A run is an object
# with a label and a prompts map; anything else is skipped rather than crashing
# the table, which is what happened when chapter 23 dropped a JSON list in here.
def is_run(d):
    return isinstance(d, dict) and "label" in d and "prompts" in d

for p in sorted(glob.glob(os.path.join(HERE, "results", "*.json"))):
    try:
        d = json.load(open(p))
        if not is_run(d):
            continue
        runs.append(d)
    except json.JSONDecodeError:
        print(f"  (skipping unreadable {os.path.basename(p)})", file=sys.stderr)

if not runs:
    sys.exit("No results yet. Run bench/run.py against an engine first.")

runs.sort(key=lambda r: (r.get("level", "z"), r.get("started", "")))
md = "--md" in sys.argv

def g(r, pid, key):
    v = r.get("prompts", {}).get(pid, {}).get(key)
    return v if v is not None else "—"

rows = []
for r in runs:
    vram = r.get("vram_peak_mb", {})
    peak = max(vram.values()) if vram else "—"
    c = r.get("concurrency", {})
    rows.append([
        r.get("level") or "—",
        r["label"],
        r.get("quant", "—"),
        f'{g(r,"short","ttft_ms")}',
        f'{g(r,"medium","decode_tps")}',
        f'{g(r,"longctx","ttft_ms")}',
        f'{c.get("16",{}).get("total_tps","—")}',
        f'{peak}',
        f'{g(r,"medium","output_sha256")}',
    ])

head = ["", "engine", "quant", "TTFT ms", "tok/s", "TTFT 2k", "tok/s @16", "VRAM MB", "hash"]
if md:
    print("| " + " | ".join(head) + " |")
    print("|" + "|".join("---" for _ in head) + "|")
    for row in rows:
        print("| " + " | ".join(str(x) for x in row) + " |")
else:
    w = [max(len(str(r[i])) for r in rows + [head]) for i in range(len(head))]
    print("  " + "  ".join(str(h).ljust(w[i]) for i, h in enumerate(head)))
    print("  " + "  ".join("-" * w[i] for i in range(len(head))))
    for row in rows:
        print("  " + "  ".join(str(x).ljust(w[i]) for i, x in enumerate(row)))

    # The correctness column. Check EVERY prompt, not one of them — an earlier
    # version compared only "medium" and announced that all engines agreed while
    # one prompt in four disagreed. A claim must be as wide as its evidence.
    #
    # And compare WITHIN a model, never across. Once the multi-card runs added
    # Llama-3-8B alongside SmolLM2-360M, comparing every run against every other
    # reported "0 of 4 identical" — which was true and meaningless, because two
    # different models are supposed to produce different text.
    by_model = {}
    for r in runs:
        by_model.setdefault(r.get("model", "?"), []).append(r)

    for model, group in sorted(by_model.items()):
        if len(group) < 2:
            print()
            print(f"  {model}: one run only — nothing to compare against.")
            continue
        print()
        print(f"  {model} — {len(group)} runs, temperature 0")
        agree, differ = [], []
        for pid in ("short", "medium", "longctx", "code"):
            seen = {}
            for r in group:
                h = g(r, pid, "output_sha256")
                if h != "—":
                    seen.setdefault(h, []).append(r.get("label", "?"))
            if len(seen) <= 1:
                agree.append(pid)
            else:
                differ.append((pid, seen))
        if not differ:
            print(f"    identical on all {len(agree)} prompts.")
        else:
            print(f"    {len(agree)} of {len(agree)+len(differ)} prompts identical.")
            for pid, seen in differ:
                print(f"    {pid:<9} {len(seen)} DISTINCT outputs:")
                for h, labels in seen.items():
                    who = ", ".join(labels)
                    print(f"      {h}  {who[:78]}")
    if len(by_model) > 1:
        print()
        print("  Compared within each model, never across — two different models")
        print("  producing different text is not a finding. A disagreement between")
        print("  runs of the SAME model is, and those are listed above.")

if "--full" in sys.argv:
    for r in runs:
        print(f"\n=== {r['label']} ({r['run_id']}) ===")
        print(json.dumps(r, indent=2))
