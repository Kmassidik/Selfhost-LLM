#!/usr/bin/env python3
"""
27_data.py — what we actually have to train on (ch.27).

Chapter 26 priced the runs: 100 million tokens is about a day, 7.2 billion is
seventy-nine. Both numbers assume the tokens exist. This counts them.

Everything here is measured with the real tokenizer rather than estimated from
file size, because chapter 14's harness already learned that characters divided
by four is wrong by 1.43x.

    uv run source/27_data.py
"""
import hashlib, json, os, sys
from collections import Counter

ROOT = "/root/Desktop/selfhostllm"
KB = "/Users/mac/Desktop/selfhostllm/knowledge-base"      # lives on the Mac
SKIP = {".git", "node_modules", "__pycache__", ".venv", "models", "results"}
TEXT = {".py", ".md", ".html", ".json", ".sh", ".txt", ".js", ".css", ".yml", ".toml"}


VENDORED = ("min.js", "min.css", ".min.", "katex", "marked", "mermaid",
            "highlight", "chart.min")


def is_vendored(path):
    """Third-party files that happen to sit in the repository are not ours."""
    base = os.path.basename(path).lower()
    return any(v in base for v in VENDORED) or "/assets/" in path.replace("\\", "/")


OURS = ("source", "bench", "docs", "planning", "experiments")


def walk(root):
    """Only directories this project wrote, plus the markdown at the top level.

    engines/ holds cloned copies of llama.cpp, vLLM and SGLang — hundreds of
    files nobody here authored. Counting them as our corpus would inflate it
    several times over.
    """
    tops = [os.path.join(root, d) for d in OURS if os.path.isdir(os.path.join(root, d))]
    for f in os.listdir(root):
        if os.path.splitext(f)[1] in TEXT:
            yield os.path.join(root, f)
    for top in tops:
        for d, dirs, files in os.walk(top):
            dirs[:] = [x for x in dirs if x not in SKIP and not x.startswith(".")]
            for f in files:
                if os.path.splitext(f)[1] in TEXT:
                    p_ = os.path.join(d, f)
                    try:
                        if os.path.getsize(p_) < 4_000_000:
                            yield p_
                    except OSError:
                        pass
    return
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x not in SKIP and not x.startswith(".")]
        for f in files:
            if os.path.splitext(f)[1] in TEXT:
                p = os.path.join(d, f)
                try:
                    if os.path.getsize(p) < 4_000_000:
                        yield p
                except OSError:
                    pass


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(f"{ROOT}/models/hf/SmolLM2-360M-Instruct")

    print("=" * 76)
    print("1 · EVERYTHING THIS PROJECT WROTE, TOKENIZED")
    print("=" * 76)
    buckets = Counter(); btok = Counter(); bfiles = Counter()
    vend_ch = [0]; vend_tok = [0]
    seen = {}
    dup_tokens = 0
    for p in walk(ROOT):
        try:
            text = open(p, errors="replace").read()
        except Exception:
            continue
        if not text.strip():
            continue
        ext = os.path.splitext(p)[1]
        if is_vendored(p):
            vend_ch[0] += len(text)
            vend_tok[0] += len(tok(text, add_special_tokens=False).input_ids)
            continue
        n = len(tok(text, add_special_tokens=False).input_ids)
        h = hashlib.sha256(text.encode()).hexdigest()
        if h in seen:
            dup_tokens += n
        else:
            seen[h] = p
        buckets[ext] += len(text); btok[ext] += n; bfiles[ext] += 1

    total_tok = sum(btok.values()); total_ch = sum(buckets.values())
    print(f"  {'kind':<8}{'files':>7}{'characters':>14}{'tokens':>12}{'chars/token':>13}")
    print("  " + "-" * 56)
    for ext, n in btok.most_common():
        print(f"  {ext:<8}{bfiles[ext]:>7}{buckets[ext]:>14,}{n:>12,}"
              f"{buckets[ext]/max(n,1):>13.2f}")
    print("  " + "-" * 56)
    print(f"  {'TOTAL':<8}{sum(bfiles.values()):>7}{total_ch:>14,}{total_tok:>12,}"
          f"{total_ch/max(total_tok,1):>13.2f}")
    print("  " + "-" * 56)
    print(f"  {'vendored':<8}{'':>7}{vend_ch[0]:>14,}{vend_tok[0]:>12,}"
          f"{vend_ch[0]/max(vend_tok[0],1):>13.2f}   <- excluded")
    print()
    print(f"  exact-duplicate content: {dup_tokens:,} tokens")
    print(f"  third-party files excluded: {vend_tok[0]:,} tokens of minified libraries,")
    print(f"  plus the whole of engines/ — cloned copies of llama.cpp, vLLM and SGLang.")
    print(f"  Counting somebody else's repository as our corpus would inflate it several")
    print(f"  times over, which the first version of this script did.")

    # ---- 2 · against what chapter 26 priced -------------------------------
    print()
    print("=" * 76)
    print("2 · AGAINST WHAT CHAPTER 26 PRICED")
    print("=" * 76)
    print(f"  {'target':<32}{'tokens needed':>16}{'we have':>12}{'short by':>12}")
    print("  " + "-" * 72)
    for label, need in (("a 100M-token fine-tune", 100_000_000),
                        ("a 1B-token run", 1_000_000_000),
                        ("Chinchilla for 360M", 7_236_422_400)):
        print(f"  {label:<32}{need:>16,}{total_tok:>12,}{need/max(total_tok,1):>11.0f}x")

    # ---- 3 · does the tokenizer like code? -------------------------------
    print()
    print("=" * 76)
    print("3 · THE TOKENIZER IS NOT NEUTRAL")
    print("=" * 76)
    samples = {
        "English prose": "The engine reads every weight for each token it produces, "
                         "and memory bandwidth sets the ceiling on how fast that can go.",
        "Python code":   "def read_header(path):\n    with open(path, 'rb') as f:\n"
                         "        return struct.unpack('<Q', f.read(8))[0]\n",
        "JSON":          '{"id": "short", "max_tokens": 32, "text": "The capital of Japan is"}',
        "indentation":   "    " * 8 + "return x\n",
    }
    print(f"  {'kind':<16}{'characters':>12}{'tokens':>9}{'chars/token':>14}")
    print("  " + "-" * 52)
    for k, v in samples.items():
        n = len(tok(v, add_special_tokens=False).input_ids)
        print(f"  {k:<16}{len(v):>12,}{n:>9}{len(v)/n:>14.2f}")
    print()
    print("  A token is not a word and not a character. English prose is the most")
    print("  efficient real content at 5.00 characters per token; code is 2.89, so the")
    print("  SAME amount of code costs nearly twice as many tokens to train on.")
    print()
    print("  Runs of indentation go the other way — 8.20 characters per token, because")
    print("  the tokenizer has learned single tokens for common runs of spaces. That is")
    print("  the opposite of what it is natural to assume, and it is why a corpus")
    print("  measured in megabytes says very little about what it costs to train on.")

    json.dump({"total_tokens": total_tok, "total_chars": total_ch,
               "by_ext": {k: {"files": bfiles[k], "chars": buckets[k], "tokens": btok[k]}
                          for k in btok}},
              open(f"{ROOT}/bench/results/corpus.json", "w"), indent=1)


if __name__ == "__main__":
    main()
